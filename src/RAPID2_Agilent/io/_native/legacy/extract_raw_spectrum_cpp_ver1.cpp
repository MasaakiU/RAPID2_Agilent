#include "extract_raw_spectrum_hpp.hpp"
#include <cstdio>
#include <cstring>
#include <cstdarg>
#include <limits>

namespace inten {

static inline void set_err(char* err, size_t err_len, const char* fmt, ...) noexcept {
    if (!err || err_len == 0) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, err_len, fmt, ap);
    va_end(ap);
    err[err_len - 1] = '\0';
}

static inline uint16_t read_le_u16(const uint8_t* p) noexcept {
    return (uint16_t)p[0]
     | ((uint16_t)p[1] << 8);
}
static inline uint32_t read_le_u32(const uint8_t* p) noexcept {
    return (uint32_t)p[0]
         | ((uint32_t)p[1] << 8)
         | ((uint32_t)p[2] << 16)
         | ((uint32_t)p[3] << 24);
}
static inline uint64_t read_le_u64(const uint8_t* p) noexcept {
    return (uint64_t)p[0]
         | ((uint64_t)p[1] << 8)
         | ((uint64_t)p[2] << 16)
         | ((uint64_t)p[3] << 24)
         | ((uint64_t)p[4] << 32)
         | ((uint64_t)p[5] << 40)
         | ((uint64_t)p[6] << 48)
         | ((uint64_t)p[7] << 56);
}

static inline double read_le_f64(const uint8_t* p) noexcept {
    uint64_t u = read_le_u64(p);
    double d;
    std::memcpy(&d, &u, sizeof(double));
    return d;
}

/*
Control-word decoding (matches the common scheme that also supports -4 cleanly):

Let v < 0 be a control word. Define x = -v (>0).
Then:
    q = x / 4
    r = x % 4
If r == 0:
    size_flag = 4
    len_zero  = q - 1
Else:
    size_flag = r
    len_zero  = q

This makes:
  v = -1,-2,-3,-4  represent "switch size to 1/2/3/4 bytes with zero-run len=0"
and larger negatives represent "zero-run + switch".
*/
static inline bool decode_control_word_i8(int8_t neg_v, uint32_t& len_zero, int32_t& size_flag) noexcept {
    if (neg_v >= 0) return false;

    uint32_t x = (uint32_t)(-(int32_t)neg_v); // widen then negate (handles -128)
    uint32_t q = x >> 2;
    uint32_t r = x & 3u;

    if (r == 0) {
        if (q == 0) return false;
        size_flag = 4;
        len_zero  = q - 1;
    } else {
        size_flag = (int32_t)r; // 1..3
        len_zero  = q;
    }
    return true;
}
static inline bool decode_control_word_i16(int16_t neg_v, uint32_t& len_zero, int32_t& size_flag) noexcept {
    if (neg_v >= 0) return false;

    uint32_t x = (uint32_t)(-(int32_t)neg_v); // widen then negate (handles -32768)
    uint32_t q = x >> 2;
    uint32_t r = x & 3u;

    if (r == 0) {
        if (q == 0) return false;
        size_flag = 4;
        len_zero  = q - 1;
    } else {
        size_flag = (int32_t)r; // 1..3
        len_zero  = q;
    }
    return true;
}
static inline bool decode_control_word_i32(int32_t neg_v, uint32_t& len_zero, int32_t& size_flag) noexcept {
    if (neg_v >= 0) return false;

    uint32_t x = (uint32_t)(-(int64_t)neg_v); // widen then negate (handles INT32_MIN)
    uint32_t q = x >> 2;
    uint32_t r = x & 3u;

    if (r == 0) {
        if (q == 0) return false;
        size_flag = 4;
        len_zero  = q - 1;
    } else {
        size_flag = (int32_t)r; // 1..3
        len_zero  = q;
    }
    return true;
}



int parse_header(
    const uint8_t* base,
    size_t buf_len,
    size_t offset,
    Header* out,
    char* err,
    size_t err_len
) noexcept {
    if (!base || !out) {
        set_err(err, err_len, "null pointer");
        return 1;
    }
    // header uses [offset : offset+28)
    if (offset + 28 > buf_len) {
        set_err(err, err_len, "buffer too small for header: offset=%zu", offset);
        return 2;
    }

    out->start_t = read_le_f64(base + offset + 0);
    out->delta_t = read_le_f64(base + offset + 8);

    uint32_t header_4 = read_le_u32(base + offset + 16);
    out->n_datapoints = header_4 & 0x00FFFFFFu;
    out->magic        = (uint8_t)(header_4 >> 24);

    if (out->magic != 0x90u) {
        set_err(err, err_len, "bad magic: got 0x%02X (expected 0x90)", (unsigned)out->magic);
        return 3;
    }

    int32_t init_zero_repeat = (int32_t)read_le_u32(base + offset + 20);
    int32_t size_flag_raw    = (int32_t)read_le_u32(base + offset + 24);

    // match your Python: init_zero_repeat, size_flag = -init_zero_repeat, -size_flag
    init_zero_repeat = -init_zero_repeat;
    size_flag_raw    = -size_flag_raw;

    if (init_zero_repeat < 0) {
        set_err(err, err_len, "init_zero_repeat < 0: %d", (int)init_zero_repeat);
        return 4;
    }
    if (size_flag_raw < 1 || size_flag_raw > 4) {
        set_err(err, err_len, "size_flag out of range: %d", (int)size_flag_raw);
        return 5;
    }

    out->init_zero_repeat = init_zero_repeat;
    out->size_flag        = size_flag_raw;
    return 0;
}

int decode_payload(
    const uint8_t* base,
    size_t buf_len,
    size_t offset,
    size_t compressed_len,
    const Header& header,
    uint32_t* out,
    size_t out_len,
    size_t* written,
    char* err,
    size_t err_len
) noexcept {
    if (!base || !out || !written) {
        set_err(err, err_len, "null pointer");
        return 1;
    }
    if (compressed_len < 28) {
        set_err(err, err_len, "compressed_len < 28: %zu", compressed_len);
        return 2;
    }
    const size_t block_end_off = offset + compressed_len;
    if (block_end_off > buf_len) {
        set_err(err, err_len, "block exceeds buffer: end=%zu buf_len=%zu", block_end_off, buf_len);
        return 3;
    }
    if (header.init_zero_repeat > (int32_t)out_len) {
        set_err(err, err_len, "init_zero_repeat (%d) > out_len (%zu)", (int)header.init_zero_repeat, out_len);
        return 4;
    }

    const uint8_t* p   = base + offset + 28;
    const uint8_t* end = base + block_end_off;

    uint32_t* outp     = out + (size_t)header.init_zero_repeat;
#ifndef NDEBUG
    uint32_t* out_endp = out + out_len;
#endif

#ifndef NDEBUG
    if (outp > out_endp) {
        set_err(err, err_len, "output pointer past end");
        return 8;
    }
#endif

    int32_t size_flag = header.size_flag;
    if (size_flag < 1 || size_flag > 4) {
        set_err(err, err_len, "invalid initial size_flag=%d", (int)size_flag);
        return 5;
    }

    // Outer loop: dispatch by current size_flag.
    // Inner loop: fixed-size decoding WITHOUT per-iteration switch.
    while (p < end) {
        bool switched = false;

        switch (size_flag) {
            case 1: { // 1 byte
                const uint8_t* limit = end - 1;
                while (p <= limit) {
                    const uint8_t u = *p++;

                    // sign-bit check: top bit 0 => data (0..127)
                    if ((u & 0x80u) == 0) {
#ifndef NDEBUG
                        if (outp >= out_endp) {
                            set_err(err, err_len, "output overflow: out_len=%zu", out_len);
                            return 8;
                        }
#endif
                        *outp++ = (uint32_t)u;
                    } else {
                        // control word (negative in int8)
                        const int8_t v = (int8_t)u;

                        uint32_t len_zero = 0;
                        int32_t next_flag = 0;
                        if (!decode_control_word_i8(v, len_zero, next_flag)) {
                            set_err(err, err_len, "invalid control word: %d", (int)v);
                            return 10;
                        }
                        if (next_flag < 1 || next_flag > 4) {
                            set_err(err, err_len, "decoded next_flag out of range: %d", (int)next_flag);
                            return 11;
                        }
#ifndef NDEBUG
                        if (outp + (size_t)len_zero > out_endp) {
                            set_err(err, err_len,
                                    "zero-run overflow: cur=%zu len_zero=%zu out_len=%zu",
                                    (size_t)(outp - out), (size_t)len_zero, out_len);
                            return 12;
                        }
#endif
                        outp += (size_t)len_zero;
                        size_flag = next_flag;
                        switched = true;
                        break; // re-dispatch
                    }
                }

                if (switched) break;

                if (p != end) {
                    set_err(err, err_len, "truncated payload: need 1 bytes, have %zu", (size_t)(end - p));
                    return 6;
                }
                break;
            }

            case 2: { // 2 bytes
                const uint8_t* limit = end - 2;
                while (p <= limit) {
                    uint16_t u;
                    std::memcpy(&u, p, 2);
                    p += 2;

                    // sign-bit check: top bit 0 => data (0..32767)
                    if ((u & 0x8000u) == 0) {
#ifndef NDEBUG
                        if (outp >= out_endp) {
                            set_err(err, err_len, "output overflow: out_len=%zu", out_len);
                            return 8;
                        }
#endif
                        *outp++ = (uint32_t)u;
                    } else {
                        // control word (negative in int16)
                        const int16_t v = (int16_t)u;

                        uint32_t len_zero = 0;
                        int32_t next_flag = 0;
                        if (!decode_control_word_i16(v, len_zero, next_flag)) {
                            set_err(err, err_len, "invalid control word: %d", (int)v);
                            return 10;
                        }
                        if (next_flag < 1 || next_flag > 4) {
                            set_err(err, err_len, "decoded next_flag out of range: %d", (int)next_flag);
                            return 11;
                        }
#ifndef NDEBUG
                        if (outp + (size_t)len_zero > out_endp) {
                            set_err(err, err_len,
                                    "zero-run overflow: cur=%zu len_zero=%zu out_len=%zu",
                                    (size_t)(outp - out), (size_t)len_zero, out_len);
                            return 12;
                        }
#endif
                        outp += (size_t)len_zero;
                        size_flag = next_flag;
                        switched = true;
                        break; // re-dispatch
                    }
                }

                if (switched) break;

                if (p != end) {
                    set_err(err, err_len, "truncated payload: need 2 bytes, have %zu", (size_t)(end - p));
                    return 6;
                }
                break;
            }

            case 3: { // 4 bytes
                const uint8_t* limit = end - 4;
                while (p <= limit) {
                    uint32_t u;
                    std::memcpy(&u, p, 4);
                    p += 4;

                    // sign-bit check: top bit 0 => data (0..2^31-1)
                    if ((u & 0x80000000u) == 0) {
#ifndef NDEBUG
                        if (outp >= out_endp) {
                            set_err(err, err_len, "output overflow: out_len=%zu", out_len);
                            return 8;
                        }
#endif
                        *outp++ = u;
                    } else {
                        // control word (negative in int32)
                        const int32_t v = (int32_t)u;

                        uint32_t len_zero = 0;
                        int32_t next_flag = 0;
                        if (!decode_control_word_i32(v, len_zero, next_flag)) {
                            set_err(err, err_len, "invalid control word: %d", (int)v);
                            return 10;
                        }
                        if (next_flag < 1 || next_flag > 4) {
                            set_err(err, err_len, "decoded next_flag out of range: %d", (int)next_flag);
                            return 11;
                        }
#ifndef NDEBUG
                        if (outp + (size_t)len_zero > out_endp) {
                            set_err(err, err_len,
                                    "zero-run overflow: cur=%zu len_zero=%zu out_len=%zu",
                                    (size_t)(outp - out), (size_t)len_zero, out_len);
                            return 12;
                        }
#endif
                        outp += (size_t)len_zero;
                        size_flag = next_flag;
                        switched = true;
                        break; // re-dispatch
                    }
                }

                if (switched) break;

                if (p != end) {
                    set_err(err, err_len, "truncated payload: need 4 bytes, have %zu", (size_t)(end - p));
                    return 6;
                }
                break;
            }



            case 4: { // 8 bytes (unexpected)
                set_err(err, err_len,
                        "unexpected 8-byte mode (size_flag=4) at input_off=%zu (consumed=%zu bytes)",
                        (size_t)(p - base), (size_t)(p - (base + offset + 28)));
                return 14; // 適当なエラーコード（未使用のもの）
            }

            default:
                set_err(err, err_len, "invalid size_flag=%d", (int)size_flag);
                return 5;
        }

        if (switched) {
            // re-dispatch with updated size_flag
            continue;
        } else {
            // We must be at end now, otherwise truncation returned above.
            break;
        }
    }

    *written = (size_t)(outp - out);
    return 0;
}
} // namespace