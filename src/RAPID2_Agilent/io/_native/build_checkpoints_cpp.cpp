// checkpoint_builder.cpp
#include "build_checkpoints_hpp.hpp"
#include <limits>

namespace inten {

int build_checkpoints_cpp(
    const uint8_t* base,
    std::size_t offset,
    std::size_t compressed_len,
    std::int32_t N_datapoints,
    std::int32_t stride_bytes,
    std::int64_t* offsets_out,
    std::int32_t* out_index_out,
    std::uint8_t* cur_size_out,
    std::int32_t max_checkpoints,
    std::int32_t* n_filled_out,
    std::int32_t* cur_idx
) noexcept {
    if (!base || !offsets_out || !out_index_out || !cur_size_out || !n_filled_out || !cur_idx) return 1;
    if (compressed_len < 24) return 2;
    if (N_datapoints < 0 || stride_bytes <= 0) return 3;
    if (max_checkpoints <= 0) return 4;

    const uint8_t* p = base + (offset + 24);
    const uint8_t* end = base + (offset + compressed_len);

    std::uint8_t cur_size = 4; // flag=3 -> 4 bytes
    std::int32_t cur_checkpoint_index = 0;
    std::int32_t accumulated_datapoints = 0;

    // checkpoint 0
    offsets_out[cur_checkpoint_index]   = (std::int64_t)(p - base);
    out_index_out[cur_checkpoint_index] = *cur_idx;
    cur_size_out[cur_checkpoint_index]  = cur_size;
    cur_checkpoint_index = 1;

    auto maybe_emit = [&]() noexcept -> int {
        if (stride_bytes < accumulated_datapoints) {
            if (cur_checkpoint_index >= max_checkpoints) return 5;
            offsets_out[cur_checkpoint_index]   = (std::int64_t)(p - base);
            out_index_out[cur_checkpoint_index] = *cur_idx;
            cur_size_out[cur_checkpoint_index]  = cur_size;
            cur_checkpoint_index += 1;
            accumulated_datapoints = 0;
        }
        return 0;
    };

    while (p < end) {
        if (cur_size == 1) {
            while (p < end) {
                uint8_t b0 = *p;
                if (b0 < 0x80) { // >=0
                    p += 1;
                    (*cur_idx) += 1;
                    accumulated_datapoints += 1;
                    int rc = maybe_emit(); if (rc) { *n_filled_out = cur_checkpoint_index; return rc; }
                    continue;
                }
                int8_t v8 = (int8_t)b0;
                p += 1;

                int32_t ctrl = -(int32_t)v8;
                int32_t len_zero = (ctrl >> 2);
                int32_t flag = (ctrl & 3);

                (*cur_idx) += len_zero;
                accumulated_datapoints += len_zero;

                if (flag != 0) {
                    cur_size = (flag == 1 ? 1 : (flag == 2 ? 2 : 4));
                    int rc = maybe_emit(); if (rc) { *n_filled_out = cur_checkpoint_index; return rc; }
                    break;
                }

                int rc = maybe_emit(); if (rc) { *n_filled_out = cur_checkpoint_index; return rc; }
            }
        } else if (cur_size == 2) {
            while (p + 2 <= end) {
                if (p[1] < 0x80) { // >=0
                    p += 2;
                    (*cur_idx) += 1;
                    accumulated_datapoints += 1;
                    int rc = maybe_emit(); if (rc) { *n_filled_out = cur_checkpoint_index; return rc; }
                    continue;
                }
                int16_t v16 = read_le_i16(p);
                p += 2;

                int32_t ctrl = (v16 == std::numeric_limits<int16_t>::min())
                                 ? (1 << 15)
                                 : -(int32_t)v16;

                int32_t len_zero = (ctrl >> 2);
                int32_t flag = (ctrl & 3);

                (*cur_idx) += len_zero;
                accumulated_datapoints += len_zero;

                if (flag != 0) {
                    cur_size = (flag == 1 ? 1 : (flag == 2 ? 2 : 4));
                    int rc = maybe_emit(); if (rc) { *n_filled_out = cur_checkpoint_index; return rc; }
                    break;
                }

                int rc = maybe_emit(); if (rc) { *n_filled_out = cur_checkpoint_index; return rc; }
            }
            if (p < end && (end - p) < 2) break;
        } else { // 4
            while (p + 4 <= end) {
                if (p[3] < 0x80) { // >=0
                    p += 4;
                    (*cur_idx) += 1;
                    accumulated_datapoints += 1;
                    int rc = maybe_emit(); if (rc) { *n_filled_out = cur_checkpoint_index; return rc; }
                    continue;
                }
                int32_t v32 = read_le_i32(p);
                p += 4;

                int32_t ctrl = (v32 == std::numeric_limits<int32_t>::min())
                                 ? (int32_t)0x80000000u
                                 : -v32;

                int32_t len_zero = (ctrl >> 2);
                int32_t flag = (ctrl & 3);

                (*cur_idx) += len_zero;
                accumulated_datapoints += len_zero;

                if (flag != 0) {
                    cur_size = (flag == 1 ? 1 : (flag == 2 ? 2 : 4));
                    int rc = maybe_emit(); if (rc) { *n_filled_out = cur_checkpoint_index; return rc; }
                    break;
                }

                int rc = maybe_emit(); if (rc) { *n_filled_out = cur_checkpoint_index; return rc; }
            }
            if (p < end && (end - p) < 4) break;
        }
    }

    *n_filled_out = cur_checkpoint_index;
    return 0;
}

} // namespace inten
