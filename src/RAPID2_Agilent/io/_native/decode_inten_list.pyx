# distutils: language=c++
# distutils: extra_compile_args = ["-O3", "-march=native", "-std=c++20", "-DNDEBUG", "-flto"]
# distutils: extra_link_args=["-flto"]
# distutils: sources = ["decode_inten_list_cpp.cpp"]
# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

# cythonize -3 -a -i decode_inten_list.pyx

from libc.stdint cimport uint8_t, uint32_t, int32_t
from libc.stddef cimport size_t

cdef extern from "decode_inten_list_hpp.hpp" namespace "inten":
    int decompress_intensity_cpp(
        const uint8_t* base,
        size_t token_offset,
        int32_t out_idx_start,
        int32_t stop_len,
        uint8_t cur_size_start,
        uint32_t* out_u32,
        size_t* out_final_offset,
        int32_t* out_idx_end
    ) nogil

def decode_inten_list(const unsigned char[::1] mm_view,
                             Py_ssize_t token_offset,
                             Py_ssize_t out_idx_start,
                             Py_ssize_t stop_len,
                             Py_ssize_t cur_size_start,
                             out):
    """
    out: np.uint32 1D C-contig (writable)
    token_offset: トークン列開始（例: offset+24 or checkpoint.offset）
    out_idx_start: outの開始位置（checkpoint.out_pos）
    stop_len: 到達したら停止（例: N_datapoints_no_end_zeros）
    cur_size_start: 1/2/4（checkpoint.cur_size）
    Returns: (final_offset:int, out_idx_end:int)
    """
    if token_offset < 0:
        raise ValueError("token_offset must be >= 0")
    if out_idx_start < 0:
        raise ValueError("out_idx_start must be >= 0")
    if stop_len < 0:
        raise ValueError("stop_len must be >= 0")
    if cur_size_start not in (1, 2, 4):
        raise ValueError("cur_size_start must be 1,2,4")

    cdef uint32_t[::1] out_mv = out

    if out_mv.shape[0] < stop_len:
        raise ValueError(f"out is too short: len(out)={out_mv.shape[0]} < stop_len={stop_len}")
    if out_idx_start > stop_len:
        raise ValueError("out_idx_start must be <= stop_len")

    cdef size_t final_off = 0
    cdef int32_t out_idx_end = 0
    cdef int rc

    with nogil:
        rc = decompress_intensity_cpp(
            <const uint8_t*>&mm_view[0],
            <size_t>token_offset,
            <int32_t>out_idx_start,
            <int32_t>stop_len,
            <uint8_t>cur_size_start,
            <uint32_t*>&out_mv[0],
            &final_off,
            &out_idx_end
        )

    if rc != 0:
        raise ValueError(f"decompress_intensity_cpp failed rc={rc}")

    return int(final_off), int(out_idx_end)
