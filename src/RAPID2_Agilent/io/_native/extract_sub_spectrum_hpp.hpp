#pragma once
#include <cstdint>
#include <cstddef>

namespace inten {

static inline int16_t read_le_i16_sub(const uint8_t* p) noexcept {
    uint16_t u = (uint16_t)p[0] | ((uint16_t)p[1] << 8);
    return (int16_t)u;
}

static inline int32_t read_le_i32_sub(const uint8_t* p) noexcept {
    uint32_t u = (uint32_t)p[0]
               | ((uint32_t)p[1] << 8)
               | ((uint32_t)p[2] << 16)
               | ((uint32_t)p[3] << 24);
    return (int32_t)u;
}

int extract_sub_spectrum_cpp(
    const uint8_t* base,
    std::size_t token_offset,
    std::int32_t out_idx_start,
    std::int32_t start_index,
    std::int32_t stop_index,
    std::uint8_t cur_size_start,
    std::uint32_t* out_u32,
    std::size_t* out_final_offset,
    std::int32_t* out_idx_end
) noexcept;

int sum_sub_spectrum_cpp(
    const uint8_t* base,
    std::size_t token_offset,
    std::int32_t out_idx_start,
    std::int32_t start_index,
    std::int32_t stop_index,
    std::uint8_t cur_size_start,
    std::uint64_t* sum_out,
    std::size_t* out_final_offset,
    std::int32_t* out_idx_end
) noexcept;

} // namespace inten
