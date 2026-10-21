# distutils: language=c++
# distutils: extra_compile_args = ["-O3", "-march=native", "-std=c++20", "-DNDEBUG", "-flto"]
# distutils: extra_link_args=["-flto"]
# distutils: sources = ["extract_sub_spectrum_cpp.cpp"]
# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

# cythonize -3 -a -i extract_sub_spectrum.pyx

from libc.stdint cimport uint8_t, uint32_t, uint64_t, int32_t
from libc.stddef cimport size_t

import numpy as np

cdef extern from "extract_sub_spectrum_hpp.hpp" namespace "inten":
    int extract_sub_spectrum_cpp(
        const uint8_t* base,
        size_t token_offset,
        int32_t out_idx_start,
        int32_t start_index,
        int32_t stop_index,
        uint8_t cur_size_start,
        uint32_t* out_u32,
        size_t* out_final_offset,
        int32_t* out_idx_end
    ) nogil
    int sum_sub_spectrum_cpp(
        const uint8_t* base,
        size_t token_offset,
        int32_t out_idx_start,
        int32_t start_index,
        int32_t stop_index,
        uint8_t cur_size_start,
        uint64_t* sum_out,
        size_t* out_final_offset,
        int32_t* out_idx_end
    ) nogil

def extract_sub_spectrum(const unsigned char[::1] mm_view,
                         Py_ssize_t token_offset,
                         Py_ssize_t out_idx_start,
                         Py_ssize_t start_index,
                         Py_ssize_t stop_index,
                         Py_ssize_t cur_size_start):
    if token_offset < 0:
        raise ValueError("token_offset must be >= 0")
    if out_idx_start < 0:
        raise ValueError("out_idx_start must be >= 0")
    if start_index < 0:
        raise ValueError("start_index must be >= 0")
    if stop_index < start_index:
        raise ValueError("stop_index must be >= start_index")
    if cur_size_start not in (1, 2, 4):
        raise ValueError("cur_size_start must be 1,2,4")

    out = np.zeros(stop_index - start_index, dtype=np.uint32)
    if stop_index == start_index:
        return out

    cdef uint32_t[::1] out_mv = out
    cdef size_t final_off = 0
    cdef int32_t out_idx_end = 0
    cdef int rc

    with nogil:
        rc = extract_sub_spectrum_cpp(
            <const uint8_t*>&mm_view[0],
            <size_t>token_offset,
            <int32_t>out_idx_start,
            <int32_t>start_index,
            <int32_t>stop_index,
            <uint8_t>cur_size_start,
            <uint32_t*>&out_mv[0],
            &final_off,
            &out_idx_end
        )

    if rc != 0:
        raise ValueError(f"extract_sub_spectrum_cpp failed rc={rc}")

    return out

def sum_sub_spectrum(const unsigned char[::1] mm_view,
                     Py_ssize_t token_offset,
                     Py_ssize_t out_idx_start,
                     Py_ssize_t start_index,
                     Py_ssize_t stop_index,
                     Py_ssize_t cur_size_start):
    if token_offset < 0:
        raise ValueError("token_offset must be >= 0")
    if out_idx_start < 0:
        raise ValueError("out_idx_start must be >= 0")
    if start_index < 0:
        raise ValueError("start_index must be >= 0")
    if stop_index < start_index:
        raise ValueError("stop_index must be >= start_index")
    if cur_size_start not in (1, 2, 4):
        raise ValueError("cur_size_start must be 1,2,4")
    if stop_index == start_index:
        return 0

    cdef uint64_t sum_out = 0
    cdef size_t final_off = 0
    cdef int32_t out_idx_end = 0
    cdef int rc

    with nogil:
        rc = sum_sub_spectrum_cpp(
            <const uint8_t*>&mm_view[0],
            <size_t>token_offset,
            <int32_t>out_idx_start,
            <int32_t>start_index,
            <int32_t>stop_index,
            <uint8_t>cur_size_start,
            &sum_out,
            &final_off,
            &out_idx_end
        )

    if rc != 0:
        raise ValueError(f"sum_sub_spectrum_cpp failed rc={rc}")

    return int(sum_out)
