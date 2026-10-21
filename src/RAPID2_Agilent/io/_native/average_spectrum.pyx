# distutils: language=c++
# distutils: extra_compile_args = ["-O3", "-march=native", "-std=c++20", "-DNDEBUG", "-flto"]
# distutils: extra_link_args=["-flto"]
# distutils: sources = ["average_spectrum_cpp.cpp"]
# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

# cythonize -3 -a -i average_spectrum.pyx

from libc.stdint cimport int32_t, uint8_t, uint32_t
from libc.stddef cimport size_t


cdef extern from "average_spectrum_hpp.hpp" namespace "spectrum":
    int accumulate_nearest_bins_cpp(
        const double* common_mz,
        size_t common_len,
        const double* mz_list,
        const uint32_t* inten_list,
        size_t mz_len,
        double* accumulated_intensity
    ) nogil
    int accumulate_decoded_calibrated_nearest_bins_cpp(
        const uint8_t* base,
        size_t token_offset,
        int32_t out_idx_start,
        int32_t stop_len,
        uint8_t cur_size_start,
        const double* common_mz,
        size_t common_len,
        double coeff,
        double base_cal,
        const double* poly,
        size_t poly_len,
        double start_t,
        double delta_t,
        double* accumulated_intensity
    ) nogil


def accumulate_nearest_bins(const double[::1] common_mz,
                            const double[::1] mz_list,
                            const uint32_t[::1] inten_list,
                            double[::1] accumulated_intensity):
    if common_mz.shape[0] == 0:
        raise ValueError("common_mz must not be empty")
    if mz_list.shape[0] != inten_list.shape[0]:
        raise ValueError("mz_list and inten_list must have the same length")
    if accumulated_intensity.shape[0] != common_mz.shape[0]:
        raise ValueError("accumulated_intensity and common_mz must have the same length")
    if mz_list.shape[0] == 0:
        return

    cdef int rc
    with nogil:
        rc = accumulate_nearest_bins_cpp(
            &common_mz[0],
            <size_t>common_mz.shape[0],
            &mz_list[0],
            &inten_list[0],
            <size_t>mz_list.shape[0],
            &accumulated_intensity[0]
        )

    if rc != 0:
        raise ValueError(f"accumulate_nearest_bins_cpp failed rc={rc}")


def accumulate_decoded_calibrated_nearest_bins(
    const unsigned char[::1] mm_view,
    Py_ssize_t token_offset,
    Py_ssize_t out_idx_start,
    Py_ssize_t stop_len,
    Py_ssize_t cur_size_start,
    const double[::1] common_mz,
    double coeff,
    double base_cal,
    const double[::1] poly,
    double start_t,
    double delta_t,
    double[::1] accumulated_intensity,
):
    if token_offset < 0:
        raise ValueError("token_offset must be >= 0")
    if out_idx_start < 0:
        raise ValueError("out_idx_start must be >= 0")
    if stop_len < 0:
        raise ValueError("stop_len must be >= 0")
    if cur_size_start not in (1, 2, 4):
        raise ValueError("cur_size_start must be 1,2,4")
    if common_mz.shape[0] == 0:
        raise ValueError("common_mz must not be empty")
    if accumulated_intensity.shape[0] != common_mz.shape[0]:
        raise ValueError("accumulated_intensity and common_mz must have the same length")

    if poly.shape[0] < stop_len:
        raise ValueError("poly must cover every decoded datapoint")
    if out_idx_start >= stop_len:
        return

    cdef int rc
    with nogil:
        rc = accumulate_decoded_calibrated_nearest_bins_cpp(
            <const uint8_t*>&mm_view[0],
            <size_t>token_offset,
            <int32_t>out_idx_start,
            <int32_t>stop_len,
            <uint8_t>cur_size_start,
            &common_mz[0],
            <size_t>common_mz.shape[0],
            coeff,
            base_cal,
            &poly[0],
            <size_t>poly.shape[0],
            start_t,
            delta_t,
            &accumulated_intensity[0],
        )

    if rc != 0:
        raise ValueError(f"accumulate_decoded_calibrated_nearest_bins_cpp failed rc={rc}")
