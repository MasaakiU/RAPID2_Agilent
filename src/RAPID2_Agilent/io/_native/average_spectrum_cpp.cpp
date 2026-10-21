#include "average_spectrum_hpp.hpp"
#include <limits>

namespace spectrum {

static inline int16_t read_le_i16_average(const std::uint8_t* p) noexcept {
    std::uint16_t u = (std::uint16_t)p[0] | ((std::uint16_t)p[1] << 8);
    return (std::int16_t)u;
}

static inline int32_t read_le_i32_average(const std::uint8_t* p) noexcept {
    std::uint32_t u = (std::uint32_t)p[0]
                    | ((std::uint32_t)p[1] << 8)
                    | ((std::uint32_t)p[2] << 16)
                    | ((std::uint32_t)p[3] << 24);
    return (std::int32_t)u;
}

static inline std::size_t advance_nearest_common_mz_index(
    const double* common_mz,
    std::size_t common_len,
    double mz_value,
    std::size_t current_index
) noexcept {
    while (current_index + 1 < common_len) {
        const double current_distance = mz_value >= common_mz[current_index]
                                      ? mz_value - common_mz[current_index]
                                      : common_mz[current_index] - mz_value;
        const double next_distance = mz_value >= common_mz[current_index + 1]
                                   ? mz_value - common_mz[current_index + 1]
                                   : common_mz[current_index + 1] - mz_value;
        if (next_distance < current_distance) {
            current_index += 1;
        } else {
            break;
        }
    }
    return current_index;
}

int accumulate_nearest_bins_cpp(
    const double* common_mz,
    std::size_t common_len,
    const double* mz_list,
    const std::uint32_t* inten_list,
    std::size_t mz_len,
    double* accumulated_intensity
) noexcept {
    if (!common_mz || !mz_list || !inten_list || !accumulated_intensity) return 1;
    if (common_len == 0) return 2;

    std::size_t nearest_index = 0;
    for (std::size_t i = 0; i < mz_len; ++i) {
        nearest_index = advance_nearest_common_mz_index(common_mz, common_len, mz_list[i], nearest_index);
        accumulated_intensity[nearest_index] += static_cast<double>(inten_list[i]);
    }

    return 0;
}

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
) noexcept {
    if (!base || !common_mz || !poly || !accumulated_intensity) return 1;
    if (common_len == 0) return 2;
    if (stop_len < 0 || out_idx_start < 0) return 3;
    if (!(cur_size_start == 1 || cur_size_start == 2 || cur_size_start == 4)) return 4;
    if (static_cast<std::size_t>(stop_len) > poly_len) return 5;
    if (out_idx_start >= stop_len) return 0;

    auto mz_at = [&](std::int32_t datapoint_index) noexcept {
        const double t_value = start_t + delta_t * static_cast<double>(datapoint_index);
        const double primary_term = coeff * (t_value - base_cal);
        return primary_term * primary_term - poly[datapoint_index];
    };

    std::size_t nearest_index = 0;
    auto add_value = [&](std::uint32_t value, std::int32_t out_idx) noexcept {
        const double mz_value = mz_at(out_idx);
        nearest_index = advance_nearest_common_mz_index(common_mz, common_len, mz_value, nearest_index);
        accumulated_intensity[nearest_index] += static_cast<double>(value);
    };

    const std::uint8_t* p = base + token_offset;
    std::uint8_t cur_size = cur_size_start;
    std::int32_t out_idx = out_idx_start;

    while (out_idx < stop_len) {
        if (cur_size == 1) {
            while (out_idx < stop_len) {
                const std::uint8_t b0 = *p;
                if (b0 < 0x80) {
                    add_value(static_cast<std::uint32_t>(b0), out_idx);
                    out_idx += 1;
                    p += 1;
                    continue;
                }

                const std::int8_t v8 = static_cast<std::int8_t>(b0);
                p += 1;

                const std::int32_t ctrl = -static_cast<std::int32_t>(v8);
                const std::int32_t len_zero = (ctrl >> 2);
                const std::int32_t flag = (ctrl & 3);

                out_idx += len_zero;
                if (out_idx >= stop_len) break;

                if (flag != 0) {
                    cur_size = (flag == 1 ? 1 : (flag == 2 ? 2 : 4));
                    break;
                }
            }
        } else if (cur_size == 2) {
            while (out_idx < stop_len) {
                if (p[1] < 0x80) {
                    const std::uint16_t u = static_cast<std::uint16_t>(p[0])
                                          | (static_cast<std::uint16_t>(p[1]) << 8);
                    add_value(static_cast<std::uint32_t>(u), out_idx);
                    out_idx += 1;
                    p += 2;
                    continue;
                }

                const std::int16_t v16 = read_le_i16_average(p);
                p += 2;

                const std::int32_t ctrl = (v16 == std::numeric_limits<std::int16_t>::min())
                                           ? (1 << 15)
                                           : -static_cast<std::int32_t>(v16);
                const std::int32_t len_zero = (ctrl >> 2);
                const std::int32_t flag = (ctrl & 3);

                out_idx += len_zero;
                if (out_idx >= stop_len) break;

                if (flag != 0) {
                    cur_size = (flag == 1 ? 1 : (flag == 2 ? 2 : 4));
                    break;
                }
            }
        } else {
            while (out_idx < stop_len) {
                if (p[3] < 0x80) {
                    const std::uint32_t u = static_cast<std::uint32_t>(p[0])
                                          | (static_cast<std::uint32_t>(p[1]) << 8)
                                          | (static_cast<std::uint32_t>(p[2]) << 16)
                                          | (static_cast<std::uint32_t>(p[3]) << 24);
                    add_value(u, out_idx);
                    out_idx += 1;
                    p += 4;
                    continue;
                }

                const std::int32_t v32 = read_le_i32_average(p);
                p += 4;

                const std::int32_t ctrl = (v32 == std::numeric_limits<std::int32_t>::min())
                                           ? static_cast<std::int32_t>(0x80000000u)
                                           : -v32;
                const std::int32_t len_zero = (ctrl >> 2);
                const std::int32_t flag = (ctrl & 3);

                out_idx += len_zero;
                if (out_idx >= stop_len) break;

                if (flag != 0) {
                    cur_size = (flag == 1 ? 1 : (flag == 2 ? 2 : 4));
                    break;
                }
            }
        }
    }

    return 0;
}

} // namespace spectrum
