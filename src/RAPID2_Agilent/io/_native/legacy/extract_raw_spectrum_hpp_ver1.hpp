#pragma once
#include <cstddef>
#include <cstdint>

namespace inten {

struct Header {
    double   start_t;
    double   delta_t;
    uint32_t n_datapoints;       // low 24 bits of header_4
    uint8_t  magic;              // high byte of header_4 (expects 0x90)
    int32_t  init_zero_repeat;   // already sign-flipped to >=0
    int32_t  size_flag;          // 1..4  (1=1B,2=2B,3=4B,4=8B)
};

// returns 0 on success, nonzero on error (err is filled)
int parse_header(
    const uint8_t* base,
    size_t buf_len,
    size_t offset,
    Header* out,
    char* err,
    size_t err_len
) noexcept;

// Decode payload into out (out must be zero-initialized).
// Returns 0 on success, nonzero on error.
// written will be the final cur_idx (typically == out_len).
int decode_payload(
    const uint8_t* base,
    size_t buf_len,
    size_t offset,
    size_t compressed_len,
    const Header& h,
    uint32_t* out,
    size_t out_len,
    size_t* written,
    char* err,
    size_t err_len
) noexcept;

} // namespace inten
