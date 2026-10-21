#include "mz_rt_binning_hpp.hpp"
#include <limits>

namespace mzrt {

static inline int16_t read_le_i16_mzrt(const std::uint8_t* p) noexcept {
    std::uint16_t u = (std::uint16_t)p[0] | ((std::uint16_t)p[1] << 8);
    return (std::int16_t)u;
}

static inline int32_t read_le_i32_mzrt(const std::uint8_t* p) noexcept {
    std::uint32_t u = (std::uint32_t)p[0]
                    | ((std::uint32_t)p[1] << 8)
                    | ((std::uint32_t)p[2] << 16)
                    | ((std::uint32_t)p[3] << 24);
    return (std::int32_t)u;
}

// Decode datapoints [start_index, stop_index) from a checkpoint and keep the
// max intensity of each bin [edges[i], edges[i+1]). The calibrated m/z grows
// with the datapoint index, so the bin index only advances.
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
) noexcept {
    if (!base || !poly || !edges || !out) return 1;
    if (n_edges < 2) return 2;
    if (stop_index < 0 || start_index < 0 || out_idx_start < 0) return 3;
    if (!(cur_size_start == 1 || cur_size_start == 2 || cur_size_start == 4)) return 4;
    if (static_cast<std::size_t>(stop_index) > poly_len) return 5;

    const double mz_first = edges[0];
    const double mz_last = edges[n_edges - 1];
    std::size_t bin = 0;
    bool beyond_last_edge = false;

    auto add_value = [&](std::uint32_t value, std::int32_t idx) noexcept {
        if (value == 0 || idx < start_index) return;
        const double t_value = start_t + delta_t * static_cast<double>(idx);
        const double primary_term = coeff * (t_value - base_cal);
        const double mz_value = primary_term * primary_term - poly[idx];
        if (mz_value < mz_first) return;
        if (mz_value >= mz_last) {
            beyond_last_edge = true;
            return;
        }
        while (mz_value >= edges[bin + 1]) bin += 1;
        const float v = static_cast<float>(value);
        if (v > out[bin]) out[bin] = v;
    };

    const std::uint8_t* p = base + token_offset;
    std::uint8_t cur_size = cur_size_start;
    std::int32_t out_idx = out_idx_start;

    while (out_idx < stop_index && !beyond_last_edge) {
        if (cur_size == 1) {
            while (out_idx < stop_index && !beyond_last_edge) {
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
                out_idx += (ctrl >> 2);
                const std::int32_t flag = (ctrl & 3);
                if (out_idx >= stop_index) break;
                if (flag != 0) {
                    cur_size = (flag == 1 ? 1 : (flag == 2 ? 2 : 4));
                    break;
                }
            }
        } else if (cur_size == 2) {
            while (out_idx < stop_index && !beyond_last_edge) {
                if (p[1] < 0x80) {
                    const std::uint16_t u = static_cast<std::uint16_t>(p[0])
                                          | (static_cast<std::uint16_t>(p[1]) << 8);
                    add_value(static_cast<std::uint32_t>(u), out_idx);
                    out_idx += 1;
                    p += 2;
                    continue;
                }
                const std::int16_t v16 = read_le_i16_mzrt(p);
                p += 2;
                const std::int32_t ctrl = (v16 == std::numeric_limits<std::int16_t>::min())
                                           ? (1 << 15)
                                           : -static_cast<std::int32_t>(v16);
                out_idx += (ctrl >> 2);
                const std::int32_t flag = (ctrl & 3);
                if (out_idx >= stop_index) break;
                if (flag != 0) {
                    cur_size = (flag == 1 ? 1 : (flag == 2 ? 2 : 4));
                    break;
                }
            }
        } else {
            while (out_idx < stop_index && !beyond_last_edge) {
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
                const std::int32_t v32 = read_le_i32_mzrt(p);
                p += 4;
                const std::int32_t ctrl = (v32 == std::numeric_limits<std::int32_t>::min())
                                           ? static_cast<std::int32_t>(0x80000000u)
                                           : -v32;
                out_idx += (ctrl >> 2);
                const std::int32_t flag = (ctrl & 3);
                if (out_idx >= stop_index) break;
                if (flag != 0) {
                    cur_size = (flag == 1 ? 1 : (flag == 2 ? 2 : 4));
                    break;
                }
            }
        }
    }

    return 0;
}

} // namespace mzrt
