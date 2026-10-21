# distutils: language=c++
# distutils: extra_compile_args = ["-O3", "-march=native", "-std=c++20", "-DNDEBUG", "-flto"]
# distutils: sources = ["build_checkpoints_cpp.cpp"]
# distutils: extra_link_args=["-flto"]
# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

# cythonize -3 -a -i build_checkpoints.pyx

from libc.stdint cimport int32_t, int64_t, uint8_t
from libc.stddef cimport size_t

import numpy as np

cdef extern from "build_checkpoints_hpp.hpp" namespace "inten":
    int build_checkpoints_cpp(
        const uint8_t* base,
        size_t offset,
        size_t compressed_len,
        int32_t N_datapoints,
        int32_t stride_bytes,
        int64_t* offsets_out,
        int32_t* out_index_out,
        uint8_t* cur_size_out,
        int32_t max_checkpoints,
        int32_t* n_filled_out, 
        int32_t* cur_idx
    ) nogil

def build_checkpoints(const unsigned char[::1] mm_view,
                           Py_ssize_t offset,
                           Py_ssize_t compressed_len,
                           Py_ssize_t N_datapoints,
                           Py_ssize_t cur_idx,
                           Py_ssize_t stride_bytes=2048):
    """
    Return:
      offsets_np(int64), out_index_np(int32), cur_size_np(uint8), n_filled(int)
    arrays are length N = ceil(N_datapoints/stride_bytes) (same as your Python allocation),
    unused tail is filled with -1/0.
    """
    cdef int32_t N
    if N_datapoints < 0:
        raise ValueError("N_datapoints must be >= 0")
    if stride_bytes <= 0:
        raise ValueError("stride_bytes must be > 0")
    if cur_idx < 0:
        raise ValueError("cur_idx must be >= 0")

    # match your Python allocation
    N = <int32_t>((N_datapoints + stride_bytes - 1) // stride_bytes)

    offsets_np = np.empty(N, dtype=np.int64)
    out_index_np = np.empty(N, dtype=np.int32)
    cur_size_np = np.empty(N, dtype=np.uint8)

    offsets_np.fill(-1)
    out_index_np.fill(-1)
    cur_size_np.fill(0)

    cdef long long[::1] offsets_mv = offsets_np
    cdef int[::1] out_index_mv = out_index_np
    cdef unsigned char[::1] cur_size_mv = cur_size_np

    cdef int32_t n_filled = 0
    cdef int32_t cur_idx_c = <int32_t>cur_idx
    cdef int rc

    with nogil:
        rc = build_checkpoints_cpp(
            <const uint8_t*>&mm_view[0],
            <size_t>offset,
            <size_t>compressed_len,
            <int32_t>N_datapoints,
            <int32_t>stride_bytes,
            <int64_t*>&offsets_mv[0],
            <int32_t*>&out_index_mv[0],
            <uint8_t*>&cur_size_mv[0],
            <int32_t>N,
            &n_filled,
            &cur_idx_c
        )

    if rc != 0:
        raise ValueError(f"build_checkpoints_cpp failed rc={rc}")

    return offsets_np, out_index_np, cur_size_np, int(n_filled), int(cur_idx_c)