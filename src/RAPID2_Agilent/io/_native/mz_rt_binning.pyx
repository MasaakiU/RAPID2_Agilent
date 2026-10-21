# distutils: language=c++
# distutils: extra_compile_args = ["-O3", "-march=native", "-std=c++20", "-DNDEBUG", "-flto"]
# distutils: extra_link_args=["-flto"]
# distutils: sources = ["mz_rt_binning_cpp.cpp"]
# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

# cythonize -3 -a -i mz_rt_binning.pyx

from libc.stdint cimport int32_t, uint8_t
from libc.stddef cimport size_t


cdef extern from "mz_rt_binning_hpp.hpp" namespace "mzrt":
    int max_into_bins_cpp(
        const uint8_t* base,
        size_t token_offset,
        int32_t out_idx_start,
        int32_t start_index,
        int32_t stop_index,
        uint8_t cur_size_start,
        double coeff,
        double base_cal,
        const double* poly,
        size_t poly_len,
        double start_t,
        double delta_t,
        const double* edges,
        size_t n_edges,
        float* out
    ) nogil


def max_into_bins(
    const unsigned char[::1] mm_view,
    Py_ssize_t token_offset,
    Py_ssize_t out_idx_start,
    Py_ssize_t start_index,
    Py_ssize_t stop_index,
    Py_ssize_t cur_size_start,
    double coeff,
    double base_cal,
    const double[::1] poly,
    double start_t,
    double delta_t,
    const double[::1] edges,
    float[::1] out,
):
    """Max intensity of the datapoints [start_index, stop_index) in each bin [edges[i], edges[i+1])."""
    if token_offset < 0 or out_idx_start < 0 or start_index < 0:
        raise ValueError("token_offset, out_idx_start and start_index must be >= 0")
    if cur_size_start not in (1, 2, 4):
        raise ValueError("cur_size_start must be 1,2,4")
    if edges.shape[0] < 2:
        raise ValueError("edges must have at least two values")
    if out.shape[0] != edges.shape[0] - 1:
        raise ValueError("out must have one value per bin")
    if poly.shape[0] < stop_index:
        raise ValueError("poly must cover every decoded datapoint")
    if stop_index <= start_index:
        return

    cdef int rc
    with nogil:
        rc = max_into_bins_cpp(
            <const uint8_t*>&mm_view[0],
            <size_t>token_offset,
            <int32_t>out_idx_start,
            <int32_t>start_index,
            <int32_t>stop_index,
            <uint8_t>cur_size_start,
            coeff,
            base_cal,
            &poly[0],
            <size_t>poly.shape[0],
            start_t,
            delta_t,
            &edges[0],
            <size_t>edges.shape[0],
            &out[0],
        )

    if rc != 0:
        raise ValueError(f"max_into_bins_cpp failed rc={rc}")
