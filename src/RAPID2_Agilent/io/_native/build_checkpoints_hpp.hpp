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

// 実装は .cpp に置く
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
) noexcept;

} // namespace inten
