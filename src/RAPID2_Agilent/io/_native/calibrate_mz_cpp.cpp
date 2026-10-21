#include "calibrate_mz_hpp.hpp"

static inline double clip_value(double value, double left, double right) {
    if (value < left) return left;
    if (value > right) return right;
    return value;
}

void calculate_poly_list(
    double left,
    double right,
    double a2,
    double b2,
    double c2,
    double d2,
    double e2,
    double f2,
    const std::uint8_t* polynomial_flag,
    std::size_t N_datapoints,
    double start_t,
    double delta_t,
    double* poly_out
) {
    // Python側のロジック：
    // coefficients = (a2,b2,c2,d2,e2,f2)
    // poly_coeffs[j] に、polynomial_flag[j]==True の順に係数を詰める
    const double coefficients[6] = {a2, b2, c2, d2, e2, f2};

    double poly_coeffs[8];
    for (int polynomial_order = 0; polynomial_order < 8; ++polynomial_order) {
        poly_coeffs[polynomial_order] = 0.0;
    }

    int coeff_idx = 0;
    for (int polynomial_order = 0; polynomial_order < 8; ++polynomial_order) {
        if (polynomial_flag[polynomial_order] && coeff_idx < 6) {
            poly_coeffs[polynomial_order] = coefficients[coeff_idx];
            coeff_idx += 1;
        }
    }

    for (std::size_t datapoint_index = 0; datapoint_index < N_datapoints; ++datapoint_index) {
        const double t_value = start_t + delta_t * static_cast<double>(datapoint_index);

        const double t_clipped = clip_value(t_value, left, right);

        // Hornerで poly(t) = sum_{ord=0..7} poly_coeffs[ord] * t^ord
        double polynomial_correction = poly_coeffs[7];
        for (int polynomial_order = 6; polynomial_order >= 0; --polynomial_order) {
            polynomial_correction = polynomial_correction * t_clipped + poly_coeffs[polynomial_order];
        }

        poly_out[datapoint_index] = polynomial_correction;
    }
}

void calibrate_mz_list(
    double coeff,
    double base,
    std::size_t N_datapoints,
    double start_t,
    double delta_t,
    const double* poly,
    double* mz_list_out
) {
    for (std::size_t i = 0; i < N_datapoints; ++i) {
        const double t_value = start_t + delta_t * static_cast<double>(i);
        const double primary_term = coeff * (t_value - base);
        mz_list_out[i] = primary_term * primary_term - poly[i];
    }
}
