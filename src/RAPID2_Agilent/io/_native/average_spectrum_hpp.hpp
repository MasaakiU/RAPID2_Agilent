#pragma once
#include <cstddef>
#include <cstdint>

namespace spectrum {

int accumulate_nearest_bins_cpp(
    const double* common_mz,
    std::size_t common_len,
    const double* mz_list,
    const std::uint32_t* inten_list,
    std::size_t mz_len,
    double* accumulated_intensity
) noexcept;

int accumulate_decoded_calibrated_nearest_bins_cpp(
    const std::uint8_t* base,
    std::size_t token_offset,
    std::int32_t out_idx_start,
    std::int32_t stop_len,
    std::uint8_t cur_size_start,
    const double* common_mz,
    std::size_t common_len,
    double coeff,
    double base_cal,
    const double* poly,
    std::size_t poly_len,
    double start_t,
    double delta_t,
    double* accumulated_intensity
) noexcept;

} // namespace spectrum
