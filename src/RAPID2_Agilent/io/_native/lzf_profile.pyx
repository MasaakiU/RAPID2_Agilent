# distutils: language=c++
# distutils: extra_compile_args = ["-O3", "-march=native", "-std=c++20", "-DNDEBUG", "-flto"]
# distutils: extra_link_args=["-flto"]
# distutils: sources = ["lzf_profile_cpp.cpp"]
# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

# cythonize -3 -a -i lzf_profile.pyx

from libc.stdint cimport uint8_t, int64_t
from libc.stddef cimport size_t
from libc.string cimport memcpy
from libcpp.vector cimport vector

import numpy as np

cdef extern from "lzf_profile_hpp.hpp" namespace "inten":
    int rewrite_lzf_records_cpp(
        const uint8_t* base,
        size_t base_len,
        const int64_t* offsets,
        const int64_t* byte_counts,
        const int64_t* uncompressed_counts,
        const uint8_t* lzf,
        size_t N_records,
        vector[uint8_t]& out,
        int64_t* offsets_out,
        int64_t* byte_counts_out,
        size_t* failed_record
    ) nogil

ERRORS = {
    2: "the record is beyond the file",
    3: "no UncompressedByteCount",
    4: "out of memory",
    12: "the LZF data end within a run",
    13: "an LZF copy from before the start, or past the uncompressed size",
    14: "the LZF data end before the uncompressed size",
    21: "the uncompressed size is not 16 bytes and int32 values",
    22: "more datapoints than the token format's header holds",
    23: "a negative intensity",
}

def rewrite_lzf_records(const unsigned char[::1] mm_view, offsets, byte_counts, uncompressed_counts, lzf):
    """
    The records of MSProfile.bin at offsets, of byte_counts bytes, in one buffer: those of lzf True, LZF compressed
    (older acquisitions, e.g. of 2022) to uncompressed_counts bytes of the doubles start_t and delta_t and then an
    int32 intensity a datapoint, written in the token format of the others; the others copied.
    Returns: (buffer: uint8 ndarray, offsets: int64 ndarray, byte_counts: int64 ndarray) of the records in it
    """
    offsets_np = np.ascontiguousarray(offsets, dtype=np.int64)
    byte_counts_np = np.ascontiguousarray(byte_counts, dtype=np.int64)
    uncompressed_np = np.ascontiguousarray(uncompressed_counts, dtype=np.int64)
    lzf_np = np.ascontiguousarray(lzf, dtype=np.uint8)
    cdef size_t N = offsets_np.shape[0]
    if not (byte_counts_np.shape[0] == uncompressed_np.shape[0] == lzf_np.shape[0] == N):
        raise ValueError("offsets, byte_counts, uncompressed_counts and lzf must have one length")
    if N == 0 or mm_view.shape[0] == 0:
        raise ValueError("no records")

    offsets_out = np.empty(N, dtype=np.int64)
    byte_counts_out = np.empty(N, dtype=np.int64)
    cdef const int64_t[::1] offsets_mv = offsets_np
    cdef const int64_t[::1] byte_counts_mv = byte_counts_np
    cdef const int64_t[::1] uncompressed_mv = uncompressed_np
    cdef const uint8_t[::1] lzf_mv = lzf_np
    cdef int64_t[::1] offsets_out_mv = offsets_out
    cdef int64_t[::1] byte_counts_out_mv = byte_counts_out
    cdef vector[uint8_t] out
    cdef size_t failed = 0
    cdef int rc
    # the LZF records come to about 0.6 of their bytes (of 2022, profile without zeros left out), the others as they are
    out.reserve(<size_t>((byte_counts_np * np.where(lzf_np, 2, 3)).sum() // 3))

    with nogil:
        rc = rewrite_lzf_records_cpp(
            <const uint8_t*>&mm_view[0],
            <size_t>mm_view.shape[0],
            &offsets_mv[0],
            &byte_counts_mv[0],
            &uncompressed_mv[0],
            &lzf_mv[0],
            N,
            out,
            &offsets_out_mv[0],
            &byte_counts_out_mv[0],
            &failed
        )

    if rc != 0:
        raise ValueError(f"rewrite_lzf_records_cpp failed rc={rc} at record {failed}: {ERRORS.get(rc, 'unknown')}")

    buffer = np.empty(out.size(), dtype=np.uint8)
    cdef unsigned char[::1] buffer_mv = buffer
    if out.size():
        memcpy(&buffer_mv[0], out.data(), out.size())
    return buffer, offsets_out, byte_counts_out
