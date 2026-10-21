#pragma once

#include <cstddef>
#include <cstdint>

namespace inten {

struct Header {
    double        start_t;
    double        delta_t;
    unsigned int  n_datapoints;
    unsigned char magic;
    int           init_zero_repeat;
    int           size_flag;  // parse_headerでは 0 (bootstrap) にする
};

// Header: ★24 bytes only [offset : offset+24)
int parse_header(
    const std::uint8_t* base,
    std::size_t buf_len,
    std::size_t offset,
    Header* out,
    char* err,
    std::size_t err_len
) noexcept;

// Wrapper: payload = [offset+24 : offset+compressed_len)
int decode_payload(
    const std::uint8_t* base,
    std::size_t buf_len,
    std::size_t offset,
    std::size_t compressed_len,
    const Header& header,
    std::uint32_t* out,
    std::size_t out_len,
    std::size_t* written,
    char* err,
    std::size_t err_len
) noexcept;

// Header-free core:
//  - size_flag_io==0 => bootstrap from payload first 4 bytes (initial control word)
//  - stop when (outp - out) >= out_stop  (may overshoot due to zero-run; OK by design)
int decode_payload_at(
    const std::uint8_t* base,
    std::size_t buf_len,
    std::size_t payload_offset,
    std::size_t payload_len,
    int init_zero_repeat,      // out上の開始位置（checkpointの out_pos）
    int* size_flag_io,         // in/out：0ならbootstrap、1..4ならそのモードから開始
    std::uint32_t* out,
    std::size_t out_len,
    std::size_t out_stop,      // ★ stop threshold (absolute out index)
    std::size_t* written,      // out全体に対する書き込み終端 index（= outp-out）
    std::size_t* consumed,     // payload内で消費したバイト数（NULL可）
    char* err,
    std::size_t err_len
) noexcept;

} // namespace inten
