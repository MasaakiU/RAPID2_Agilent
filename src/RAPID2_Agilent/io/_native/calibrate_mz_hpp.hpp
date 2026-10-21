#pragma once
#include <cstddef>
#include <cstdint>

void calculate_poly_list(
    double left,
    double right,
    double a2,
    double b2,
    double c2,
    double d2,
    double e2,
    double f2,
    const std::uint8_t* polynomial_flag,   // 長さ8（0/1）
    std::size_t N_datapoints,
    double start_t,
    double delta_t,
    double* poly_out                       // 長さ N_datapoints
);

void calibrate_mz_list(
    double coeff,
    double base,
    std::size_t N_datapoints,
    double start_t,
    double delta_t,
    const double* poly,
    double* mz_list_out
);
