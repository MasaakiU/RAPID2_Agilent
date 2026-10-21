# distutils: language=c++
# distutils: extra_compile_args = ["-O3", "-march=native", "-std=c++20", "-DNDEBUG", "-flto"]
# distutils: extra_link_args=["-flto"]
# distutils: sources = ["extract_raw_spectrum_cpp.cpp"]
# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

# cythonize -3 -a -i extract_raw_spectrum.pyx

import numpy as np
from libc.stddef cimport size_t
from libc.stdint cimport uint8_t, uint32_t
from libc.stdlib cimport malloc, free

cdef extern from "extract_raw_spectrum_hpp.hpp" namespace "inten":
    cdef cppclass Header:
        double   start_t
        double   delta_t
        unsigned int n_datapoints
        unsigned char magic
        int      init_zero_repeat
        int      size_flag

    cdef struct Checkpoint:
        uint32_t out_pos
        uint32_t payload_consumed
        uint8_t  size_flag

    int parse_header(const uint8_t* base, size_t buf_len, size_t offset,
                     Header* out, char* err, size_t err_len) nogil

    # ★ out を int32_t* にした版
    int decode_payload(const uint8_t* base, size_t buf_len, size_t offset,
                       size_t compressed_len, const Header& header,
                       uint32_t* out, size_t out_len, size_t* written,
                       char* err, size_t err_len) nogil

    int build_checkpoints_by_outstride(
        const uint8_t* base,
        size_t buf_len,
        size_t payload_offset,
        size_t payload_len,
        int init_zero_repeat,
        uint32_t n_datapoints,
        size_t out_stride_bytes,
        Checkpoint* cps,
        size_t cps_cap,
        size_t* cps_count,
        char* err,
        size_t err_len
    ) noexcept



cdef inline str _cstr_to_py(char* p):
    # p must be null-terminated
    return (<bytes>p).decode("utf-8", "replace")


def decompress_intensity(buf, offset, compressed_len, out=None):
    """
    buf: memoryview(mmap) / bytes / mmap 等の buffer object
    offset: ヘッダー開始点
    compressed_len: ヘッダー+payload の総バイト長
    out: (optional) np.ndarray int32 1D contiguous (len>=N)
    """
    cdef const uint8_t[::1] view = buf
    cdef size_t buf_len = <size_t>view.shape[0]

    if offset < 0 or compressed_len < 0:
        raise ValueError("offset/compressed_len must be non-negative")

    cdef size_t off  = <size_t>offset
    cdef size_t clen = <size_t>compressed_len
    cdef const uint8_t* base = &view[0]

    cdef Header header
    cdef char err[256]
    cdef int st

    # header parse (GILあってもOKだが、関数はnogilでも呼べる)
    st = parse_header(base, buf_len, off, &header, &err[0], 256)
    if st != 0:
        raise ValueError(_cstr_to_py(&err[0]))

    cdef int N = <int>header.n_datapoints
    if N <= 0:
        raise ValueError(f"bad N_datapoints={N}")

    # Pythonで配列確保（numpyヘッダ不要）
    if out is None:
        arr = np.zeros(N, dtype=np.uint32)
    else:
        arr = np.asarray(out)
        if arr.ndim != 1:
            raise ValueError("out must be 1D")
        if arr.dtype != np.uint32:
            raise ValueError("out dtype must be uint32 (use view(np.uint32) if needed)")
        if not arr.flags["C_CONTIGUOUS"]:
            raise ValueError("out must be C-contiguous")
        if arr.shape[0] < N:
            raise ValueError(f"out too small: len(out)={arr.shape[0]} < N={N}")
        arr = arr[:N]
        arr.fill(0)

    # typed memoryviewで生ポインタを取る（numpy C-API不要）
    cdef uint32_t[::1] out_mv = arr
    cdef uint32_t* out_ptr = &out_mv[0]

    cdef size_t written = 0

    with nogil:
        st = decode_payload(base, buf_len, off, clen, header,
                            out_ptr, <size_t>N, &written,
                            &err[0], 256)

    if st != 0:
        raise ValueError(_cstr_to_py(&err[0]))
    if written != <size_t>N:
        raise ValueError(f"decoded length mismatch: written={written} expected={N}")

    return header.start_t, header.delta_t, arr

def build_checkpoints(buf, size_t offset, size_t compressed_len, size_t stride_bytes=2048):
    cdef const uint8_t[::1] view = buf
    if view.shape[0] == 0:
        raise ValueError("empty buffer")

    cdef const uint8_t* base = <const uint8_t*>&view[0]
    cdef size_t buf_len = <size_t>view.shape[0]

    cdef char err[256]
    err[0] = 0

    cdef Header hdr
    cdef int rc = parse_header(base, buf_len, offset, &hdr, err, 256)
    if rc != 0:
        raise ValueError(f"parse_header failed rc={rc}: {(<bytes>err).decode('utf-8', 'ignore')}")

    if compressed_len < 24:
        raise ValueError("compressed_len < 24")
    if offset + compressed_len > buf_len:
        raise ValueError("block exceeds buffer")

    cdef size_t payload_offset = offset + 24
    cdef size_t payload_len = compressed_len - 24

    # --- checkpoint capacity estimate ---
    # 目安: ceil(n_datapoints*4 / stride) + 少し余裕
    if stride_bytes == 0:
        raise ValueError("stride_bytes=0")

    cdef unsigned long long total_bytes = <unsigned long long>hdr.n_datapoints * 4ULL
    cdef size_t cps_cap = <size_t>((total_bytes + stride_bytes - 1) // stride_bytes + 16)

    cdef Checkpoint* cps = <Checkpoint*>malloc(cps_cap * sizeof(Checkpoint))
    if cps == NULL:
        raise MemoryError("malloc failed for checkpoints")

    cdef size_t cps_count = 0
    err[0] = 0
    rc = build_checkpoints_by_outstride(
        base, buf_len,
        payload_offset, payload_len,
        hdr.init_zero_repeat,
        hdr.n_datapoints,
        stride_bytes,
        cps, cps_cap, &cps_count,
        err, 256
    )
    if rc != 0:
        free(cps)
        raise ValueError(f"build_checkpoints failed rc={rc}: {(<bytes>err).decode('utf-8', 'ignore')}")

    # cps_count が確定した後で ndarray を作る
    out_pos = np.empty(cps_count, dtype=np.uint32)
    payload = np.empty(cps_count, dtype=np.uint32)
    flag    = np.empty(cps_count, dtype=np.uint8)

    # C-numpy API不要：buffer → typed memoryview で受ける
    cdef uint32_t[::1] out_mv = out_pos
    cdef uint32_t[::1] pay_mv = payload
    cdef uint8_t[::1]  flg_mv = flag

    cdef size_t i
    for i in range(cps_count):
        out_mv[i] = cps[i].out_pos
        pay_mv[i] = cps[i].payload_consumed
        flg_mv[i] = cps[i].size_flag

    meta = {...}
    return {"out_pos": out_pos, "payload": payload, "flag": flag, "meta": meta}
