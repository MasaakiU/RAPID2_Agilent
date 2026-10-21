# distutils: language=c++
# distutils: extra_compile_args = ["-O3", "-march=native", "-std=c++20", "-DNDEBUG", "-flto"]
# distutils: extra_link_args=["-flto"]
# distutils: sources = ["calibrate_mz_cpp.cpp"]
# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

# cythonize -3 -a -i calibrate_mz.pyx

from libc.stddef cimport size_t
import numpy as np  # Pythonレベルだけ（cimport numpy はしない）

cdef extern from "calibrate_mz_hpp.hpp":
    void calculate_poly_list(
        double left,
        double right,
        double a2,
        double b2,
        double c2,
        double d2,
        double e2,
        double f2,
        const unsigned char* polynomial_flag,  # uint8_t互換
        size_t N_datapoints,
        double start_t,
        double delta_t,
        double* poly_out
    ) nogil
    void calibrate_mz_list(
        double coeff,
        double base,
        size_t N_datapoints,
        double start_t,
        double delta_t,
        const double* poly,
        double* mz_list_out
    ) nogil


cpdef object calculate_poly(
    double left,
    double right,
    double a2,
    double b2,
    double c2,
    double d2,
    double e2,
    double f2,
    object polynomial_flag,         # Pythonのlist[bool]等（長さ8）
    int N_datapoints,
    double start_t,
    double delta_t
):
    """
    戻り値: numpy.ndarray(float64, shape=(N_datapoints,))
    cimport numpy 不使用（typed memoryviewでバッファだけ借りる）
    """

    if N_datapoints <= 0:
        raise ValueError("N_datapoints must be positive")

    # poly flag を 0/1 の 8 bytes に詰める（NumPy変換も不要）
    cdef unsigned char polynomial_flag_u8[8]
    cdef int polynomial_order

    # 長さチェック
    try:
        if len(polynomial_flag) != 8:
            raise ValueError("polynomial_flag must have length 8")
    except TypeError:
        raise ValueError("polynomial_flag must be a sized sequence (length 8)")

    for polynomial_order in range(8):
        polynomial_flag_u8[polynomial_order] = 1 if polynomial_flag[polynomial_order] else 0

    poly = np.empty(N_datapoints, dtype=np.float64)
    cdef double[::1] poly_view = poly

    with nogil:
        calculate_poly_list(
            left, right,
            a2, b2, c2, d2, e2, f2,
            &polynomial_flag_u8[0],
            <size_t>N_datapoints,
            start_t, delta_t,
            &poly_view[0]
        )

    return poly


cpdef object calibrate_mz(
    double coeff,
    double base,
    int N_datapoints,
    double start_t,
    double delta_t,
    const double[::1] poly,
):
    """Apply the per-scan primary calibration and a precomputed correction."""
    if N_datapoints <= 0:
        raise ValueError("N_datapoints must be positive")
    if poly.shape[0] != N_datapoints:
        raise ValueError("poly must have N_datapoints elements")

    mz_list = np.empty(N_datapoints, dtype=np.float64)
    cdef double[::1] mz_list_view = mz_list
    with nogil:
        calibrate_mz_list(
            coeff, base, <size_t>N_datapoints, start_t, delta_t,
            &poly[0], &mz_list_view[0]
        )
    return mz_list
