#include "decode_inten_list_hpp.hpp"
#include <limits>

namespace inten {

int decompress_intensity_cpp(
    const uint8_t* base,
    std::size_t token_offset,
    std::int32_t out_idx_start,
    std::int32_t stop_len,
    std::uint8_t cur_size_start,
    std::uint32_t* out_u32,
    std::size_t* out_final_offset,
    std::int32_t* out_idx_end
) noexcept {
    if (!base || !out_u32 || !out_final_offset || !out_idx_end) return 1;
    if (stop_len < 0) return 2;
    if (out_idx_start < 0) return 3;
    if (out_idx_start >= stop_len) {
        *out_final_offset = token_offset;
        *out_idx_end = out_idx_start;
        return 0;
    }
    if (!(cur_size_start == 1 || cur_size_start == 2 || cur_size_start == 4)) return 4;

    const uint8_t* p = base + token_offset;
    std::uint8_t cur_size = cur_size_start;
    std::int32_t out_idx  = out_idx_start;

    while (out_idx < stop_len) {
        if (cur_size == 1) {
            while (out_idx < stop_len) {
                uint8_t b0 = *p;

                if (b0 < 0x80) { // >=0
                    out_u32[out_idx] = (std::uint32_t)b0;
                    out_idx += 1;
                    p += 1;
                    continue;
                }

                int8_t v8 = (int8_t)b0;
                p += 1;

                int32_t ctrl = -(int32_t)v8;
                int32_t len_zero = (ctrl >> 2);
                int32_t flag     = (ctrl & 3);

                out_idx += len_zero;
                if (out_idx >= stop_len) break;

                if (flag != 0) {
                    cur_size = (flag == 1 ? 1 : (flag == 2 ? 2 : 4));
                    break;
                }
            }
        } else if (cur_size == 2) {
            while (out_idx < stop_len) {
                if (p[1] < 0x80) { // >=0
                    uint16_t u = (uint16_t)p[0] | ((uint16_t)p[1] << 8);
                    out_u32[out_idx] = (std::uint32_t)u;
                    out_idx += 1;
                    p += 2;
                    continue;
                }

                int16_t v16 = read_le_i16(p);
                p += 2;

                int32_t ctrl = (v16 == std::numeric_limits<int16_t>::min())
                                 ? (1 << 15)
                                 : -(int32_t)v16;

                int32_t len_zero = (ctrl >> 2);
                int32_t flag     = (ctrl & 3);

                out_idx += len_zero;
                if (out_idx >= stop_len) break;

                if (flag != 0) {
                    cur_size = (flag == 1 ? 1 : (flag == 2 ? 2 : 4));
                    break;
                }
            }
        } else { // cur_size == 4
            while (out_idx < stop_len) {
                if (p[3] < 0x80) { // >=0
                    uint32_t u = (uint32_t)p[0]
                               | ((uint32_t)p[1] << 8)
                               | ((uint32_t)p[2] << 16)
                               | ((uint32_t)p[3] << 24);
                    out_u32[out_idx] = u;
                    out_idx += 1;
                    p += 4;
                    continue;
                }

                int32_t v32 = read_le_i32(p);
                p += 4;

                int32_t ctrl = (v32 == std::numeric_limits<int32_t>::min())
                                 ? (int32_t)0x80000000u
                                 : -v32;

                int32_t len_zero = (ctrl >> 2);
                int32_t flag     = (ctrl & 3);

                out_idx += len_zero;
                if (out_idx >= stop_len) break;

                if (flag != 0) {
                    cur_size = (flag == 1 ? 1 : (flag == 2 ? 2 : 4));
                    break;
                }
            }
        }
    }

    *out_final_offset = (std::size_t)(p - base);
    *out_idx_end = out_idx;
    return 0;
}

} // namespace inten
