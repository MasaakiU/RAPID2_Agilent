#include "lzf_profile_hpp.hpp"
#include <cstring>
#include <new>

namespace inten {

namespace {

std::int32_t value_at(const std::uint8_t* values, std::size_t index) noexcept {
    const std::uint8_t* p = values + 4 * index;
    return (std::int32_t)((std::uint32_t)p[0]
                        | ((std::uint32_t)p[1] << 8)
                        | ((std::uint32_t)p[2] << 16)
                        | ((std::uint32_t)p[3] << 24));
}

// the bytes of the token of a value >= 0, and the flag that switches to them
int size_of(std::int32_t value) noexcept { return value < 0x80 ? 1 : (value < 0x8000 ? 2 : 4); }
int flag_of(int size) noexcept { return size == 1 ? 1 : (size == 2 ? 2 : 3); }
// the largest zeros * 4 + flag a negative token of the bytes holds
std::int64_t max_control(int size) noexcept { return size == 1 ? 0x7f : (size == 2 ? 0x7fff : 0x7fffffff); }

void put(std::vector<std::uint8_t>& out, std::int64_t token, int size) {
    const std::uint32_t bits = (std::uint32_t)token;
    for (int k = 0; k < size; ++k) out.push_back((std::uint8_t)(bits >> (8 * k)));
}

} // namespace

int lzf_decompress_cpp(
    const std::uint8_t* in,
    std::size_t in_len,
    std::uint8_t* out,
    std::size_t out_len
) noexcept {
    if (!in || (!out && out_len)) return 1;
    std::size_t i = 0, o = 0;
    while (i < in_len) {
        const std::size_t control = in[i++];
        if (control < 32) {  // control + 1 bytes as they are
            const std::size_t length = control + 1;
            if (length > in_len - i || length > out_len - o) return 2;
            std::memcpy(out + o, in + i, length);
            i += length;
            o += length;
            continue;
        }
        // a copy of the bytes back by the low 5 bits and the next byte + 1, of the top 3 bits + 2 (7: a byte more)
        std::size_t length = control >> 5;
        if (length == 7) {
            if (i >= in_len) return 2;
            length += in[i++];
        }
        length += 2;
        if (i >= in_len) return 2;
        const std::size_t back = ((control & 0x1f) << 8) + in[i++] + 1;
        if (back > o || length > out_len - o) return 3;
        for (std::size_t k = 0; k < length; ++k, ++o) out[o] = out[o - back];  // may overlap what it writes
    }
    return o == out_len ? 0 : 4;
}

int encode_record_cpp(
    const std::uint8_t* raw,
    std::size_t raw_len,
    std::vector<std::uint8_t>& out
) {
    if (raw_len < 16 || (raw_len - 16) % 4 != 0) return 1;
    const std::size_t N_datapoints = (raw_len - 16) / 4;
    if (N_datapoints > 0x00FFFFFF) return 2;
    const std::uint8_t* values = raw + 16;
    std::size_t first = 0;
    while (first < N_datapoints && value_at(values, first) == 0) ++first;
    std::size_t last = N_datapoints;
    while (last > first && value_at(values, last - 1) == 0) --last;

    out.insert(out.end(), raw, raw + 16);
    put(out, 0x90000000LL | (std::int64_t)N_datapoints, 4);
    put(out, -(std::int64_t)first, 4);
    int size = 4;  // of the first token
    std::size_t index = first;
    while (index < last) {
        const std::int32_t value = value_at(values, index);
        if (value < 0) return 3;
        if (value == 0) {
            std::size_t next = index;
            while (value_at(values, next) == 0) ++next;  // before last, as the value before it is not 0
            std::int64_t zeros = (std::int64_t)(next - index);
            const std::int32_t next_value = value_at(values, next);
            if (next_value < 0) return 3;
            const int next_size = size_of(next_value);
            if (4 * zeros + 3 > max_control(size)) {
                // more zeros than the token holds: as many as it does, and on in 4 bytes
                const std::int64_t held = (max_control(size) - 3) / 4;
                put(out, -(4 * held + 3), size);
                zeros -= held;
                size = 4;
            }
            put(out, -(4 * zeros + flag_of(next_size)), size);
            size = next_size;
            index = next;
            continue;
        }
        const int needed = size_of(value);
        if (needed > size) {
            put(out, -flag_of(needed), size);
            size = needed;
        } else if (needed < size) {
            // fewer bytes, if the values on that fit them save more than the switch costs
            std::int64_t saved = 0;
            for (std::size_t k = index; k < last && saved <= size; ++k) {
                const std::int32_t following = value_at(values, k);
                if (following <= 0 || size_of(following) > needed) break;
                saved += size - needed;
            }
            if (saved > size) {
                put(out, -flag_of(needed), size);
                size = needed;
            }
        }
        put(out, value, size);
        ++index;
    }
    return 0;
}

int rewrite_lzf_records_cpp(
    const std::uint8_t* base,
    std::size_t base_len,
    const std::int64_t* offsets,
    const std::int64_t* byte_counts,
    const std::int64_t* uncompressed_counts,
    const std::uint8_t* lzf,
    std::size_t N_records,
    std::vector<std::uint8_t>& out,
    std::int64_t* offsets_out,
    std::int64_t* byte_counts_out,
    std::size_t* failed_record
) noexcept {
    if (!base || !offsets || !byte_counts || !uncompressed_counts || !lzf || !offsets_out || !byte_counts_out
        || !failed_record) return 1;
    try {
        std::vector<std::uint8_t> raw;
        for (std::size_t record = 0; record < N_records; ++record) {
            *failed_record = record;
            const std::int64_t offset = offsets[record];
            const std::int64_t byte_count = byte_counts[record];
            if (offset < 0 || byte_count < 0 || (std::uint64_t)(offset + byte_count) > base_len) return 2;
            const std::uint8_t* p = base + offset;
            offsets_out[record] = (std::int64_t)out.size();
            if (lzf[record]) {
                if (uncompressed_counts[record] <= 0) return 3;
                raw.resize((std::size_t)uncompressed_counts[record]);
                int rc = lzf_decompress_cpp(p, (std::size_t)byte_count, raw.data(), raw.size());
                if (rc != 0) return 10 + rc;
                rc = encode_record_cpp(raw.data(), raw.size(), out);
                if (rc != 0) return 20 + rc;
            } else {
                out.insert(out.end(), p, p + byte_count);
            }
            byte_counts_out[record] = (std::int64_t)out.size() - offsets_out[record];
        }
    } catch (const std::bad_alloc&) {
        return 4;
    }
    return 0;
}

} // namespace inten
