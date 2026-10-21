#pragma once
#include <cstddef>
#include <cstdint>

namespace mzrt {

int max_into_bins_cpp(
    const std::uint8_t* base,
    std::size_t token_offset,
    std::int32_t out_idx_start,
    std::int32_t start_index,
    std::int32_t stop_index,
    std::uint8_t cur_size_start,
    double coeff,
    double base_cal,
    const double* poly,
    std::size_t poly_len,
    double start_t,
    double delta_t,
    const double* edges,
    std::size_t n_edges,
    float* out
) noexcept;

} // namespace mzrt
