#include "extract_raw_spectrum_cpp.hpp"
#include <cstdio>
#include <cstring>
#include <cstdarg>
#include <limits>

namespace inten {

static inline void set_err(char* err, size_t err_len, const char* fmt, ...) noexcept {
    if (!err || err_len == 0) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, err_len, fmt, ap);
    va_end(ap);
    err[err_len - 1] = '\0';
}

static inline uint16_t read_le_u16(const uint8_t* p) noexcept {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}
static inline uint32_t read_le_u32(const uint8_t* p) noexcept {
    return (uint32_t)p[0]
         | ((uint32_t)p[1] << 8)
         | ((uint32_t)p[2] << 16)
         | ((uint32_t)p[3] << 24);
}
static inline uint64_t read_le_u64(const uint8_t* p) noexcept {
    return (uint64_t)p[0]
         | ((uint64_t)p[1] << 8)
         | ((uint64_t)p[2] << 16)
         | ((uint64_t)p[3] << 24)
         | ((uint64_t)p[4] << 32)
         | ((uint64_t)p[5] << 40)
         | ((uint64_t)p[6] << 48)
         | ((uint64_t)p[7] << 56);
}

static inline int64_t read_le_i8(const uint8_t* p) noexcept {
    int8_t v = (int8_t)p[0];
    return (int64_t)v;
}
static inline int64_t read_le_i16(const uint8_t* p) noexcept {
    int16_t v = (int16_t)read_le_u16(p);
    return (int64_t)v;
}
static inline int64_t read_le_i32(const uint8_t* p) noexcept {
    int32_t v = (int32_t)read_le_u32(p);
    return (int64_t)v;
}
static inline int64_t read_le_i64(const uint8_t* p) noexcept {
    int64_t v = (int64_t)read_le_u64(p);
    return v;
}

static inline double read_le_f64(const uint8_t* p) noexcept {
    uint64_t u = read_le_u64(p);
    double d;
    std::memcpy(&d, &u, sizeof(double));
    return d;
}

static inline int size_from_flag(int32_t flag) noexcept {
    // flag: 1..4
    // size: 1,2,4,8
    switch (flag) {
        case 1: return 1;
        case 2: return 2;
        case 3: return 4;
        case 4: return 8;
        default: return 0;
    }
}

/*
Control-word decoding (matches the common scheme that also supports -4 cleanly):

Let v < 0 be a control word. Define x = -v (>0).
Then:
    q = x / 4
    r = x % 4
If r == 0:
    size_flag = 4
    len_zero  = q - 1
Else:
    size_flag = r
    len_zero  = q

This makes:
  v = -1,-2,-3,-4  represent "switch size to 1/2/3/4 bytes with zero-run len=0"
and larger negatives represent "zero-run + switch".
*/
static inline bool decode_control_word(int64_t neg_v, uint64_t& len_zero, int32_t& size_flag) noexcept {
    if (neg_v >= 0) return false;
    if (neg_v == std::numeric_limits<int64_t>::min()) return false;
    uint64_t x = (uint64_t)(-neg_v);
    uint64_t q = x >> 2;     // /4
    uint64_t r = x & 3u;     // %4

    if (r == 0) {
        // size_flag=4, len_zero=q-1
        if (q == 0) return false;
        size_flag = 4;
        len_zero  = q - 1;
    } else {
        size_flag = (int32_t)r;  // 1..3
        len_zero  = q;
    }
    return true;
}

int parse_header(
    const uint8_t* base,
    size_t buf_len,
    size_t offset,
    Header* out,
    char* err,
    size_t err_len
) noexcept {
    if (!base || !out) {
        set_err(err, err_len, "null pointer");
        return 1;
    }
    // header uses [offset : offset+28)
    if (offset + 28 > buf_len) {
        set_err(err, err_len, "buffer too small for header: offset=%zu", offset);
        return 2;
    }

    out->start_t = read_le_f64(base + offset + 0);
    out->delta_t = read_le_f64(base + offset + 8);

    uint32_t header_4 = read_le_u32(base + offset + 16);
    out->n_datapoints = header_4 & 0x00FFFFFFu;
    out->magic        = (uint8_t)(header_4 >> 24);

    if (out->magic != 0x90u) {
        set_err(err, err_len, "bad magic: got 0x%02X (expected 0x90)", (unsigned)out->magic);
        return 3;
    }

    int32_t init_zero_repeat = (int32_t)read_le_u32(base + offset + 20);
    int32_t size_flag_raw    = (int32_t)read_le_u32(base + offset + 24);

    // match your Python: init_zero_repeat, size_flag = -init_zero_repeat, -size_flag
    init_zero_repeat = -init_zero_repeat;
    size_flag_raw    = -size_flag_raw;

    if (init_zero_repeat < 0) {
        set_err(err, err_len, "init_zero_repeat < 0: %d", (int)init_zero_repeat);
        return 4;
    }
    if (size_flag_raw < 1 || size_flag_raw > 4) {
        set_err(err, err_len, "size_flag out of range: %d", (int)size_flag_raw);
        return 5;
    }

    out->init_zero_repeat = init_zero_repeat;
    out->size_flag        = size_flag_raw;
    return 0;
}

int decode_payload(
    const uint8_t* base,
    size_t buf_len,
    size_t offset,
    size_t compressed_len,
    const Header& h,
    uint32_t* out,
    size_t out_len,
    size_t* written,
    char* err,
    size_t err_len
) noexcept {
    if (!base || !out || !written) {
        set_err(err, err_len, "null pointer");
        return 1;
    }
    if (compressed_len < 28) {
        set_err(err, err_len, "compressed_len < 28: %zu", compressed_len);
        return 2;
    }
    size_t block_end = offset + compressed_len;
    if (block_end > buf_len) {
        set_err(err, err_len, "block exceeds buffer: end=%zu buf_len=%zu", block_end, buf_len);
        return 3;
    }
    if (h.init_zero_repeat > (int32_t)out_len) {
        set_err(err, err_len, "init_zero_repeat (%d) > out_len (%zu)", (int)h.init_zero_repeat, out_len);
        return 4;
    }

    const uint8_t* p   = base + offset + 28;
    const uint8_t* end = base + block_end;

    size_t cur_idx = (size_t)h.init_zero_repeat;
    int32_t size_flag = h.size_flag;
    int cur_size = size_from_flag(size_flag);

    while (p < end) {
        if (cur_size == 0) {
            set_err(err, err_len, "invalid size_flag=%d", (int)size_flag);
            return 5;
        }
        if ((size_t)(end - p) < (size_t)cur_size) {
            set_err(err, err_len, "truncated payload: need %d bytes, have %zu", cur_size, (size_t)(end - p));
            return 6;
        }

        int64_t v = 0;
        switch (cur_size) {
            case 1: v = read_le_i8(p);  break;
            case 2: v = read_le_i16(p); break;
            case 4: v = read_le_i32(p); break;
            case 8: v = read_le_i64(p); break;
            default:
                set_err(err, err_len, "unexpected cur_size=%d", cur_size);
                return 7;
        }
        p += cur_size;

        if (v >= 0) {
            if (cur_idx >= out_len) {
                set_err(err, err_len, "output overflow: cur_idx=%zu out_len=%zu", cur_idx, out_len);
                return 8;
            }
            // Store as uint32 bit-pattern; Python side can view as uint32 if desired.
            if ((int64_t)v > std::numeric_limits<int32_t>::max()) {
                set_err(err, err_len, "value too large for int32: %lld", (long long)v);
                return 9;
            }
            out[cur_idx] = (int32_t)v;
            cur_idx += 1;
        } else {
            uint64_t len_zero = 0;
            int32_t next_flag = 0;
            if (!decode_control_word(v, len_zero, next_flag)) {
                set_err(err, err_len, "invalid control word: %lld", (long long)v);
                return 10;
            }
            if (next_flag < 1 || next_flag > 4) {
                set_err(err, err_len, "decoded next_flag out of range: %d", (int)next_flag);
                return 11;
            }
            if (cur_idx + (size_t)len_zero > out_len) {
                set_err(err, err_len, "zero-run overflow: cur_idx=%zu len_zero=%zu out_len=%zu",
                        cur_idx, (size_t)len_zero, out_len);
                return 12;
            }
            cur_idx += (size_t)len_zero;
            size_flag = next_flag;
            cur_size  = size_from_flag(size_flag);
        }
    }

    *written = cur_idx;
    return 0;
}

} // namespace inten
