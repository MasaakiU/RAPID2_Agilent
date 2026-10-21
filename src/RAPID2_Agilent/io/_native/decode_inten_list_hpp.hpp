#pragma once
#include <cstdint>
#include <cstddef>

namespace inten {

static inline int16_t read_le_i16(const uint8_t* p) noexcept {
    uint16_t u = (uint16_t)p[0] | ((uint16_t)p[1] << 8);
    return (int16_t)u;
}
static inline int32_t read_le_i32(const uint8_t* p) noexcept {
    uint32_t u = (uint32_t)p[0]
               | ((uint32_t)p[1] << 8)
               | ((uint32_t)p[2] << 16)
               | ((uint32_t)p[3] << 24);
    return (int32_t)u;
}

// token_offset: トークン列の開始位置（あなたの cur_offset をそのまま渡す）
// out_idx_start: out のどこから埋めるか（= cur_idx）
// stop_len: out_idx がここに到達したら停止（外側/内側とも stop_len のみで停止）
// cur_size_start: 1/2/4（checkpoint から復帰するなら必要）
// out_idx_end: 終了時の out_idx（= どこまで進んだか）
int decompress_intensity_cpp(
    const uint8_t* base,
    std::size_t token_offset,
    std::int32_t out_idx_start,
    std::int32_t stop_len,
    std::uint8_t cur_size_start,
    std::uint32_t* out_u32,
    std::size_t* out_final_offset,
    std::int32_t* out_idx_end
) noexcept;

} // namespace inten
