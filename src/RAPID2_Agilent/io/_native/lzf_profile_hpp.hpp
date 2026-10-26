#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>

namespace inten {

// LZF (liblzf's format) of in_len bytes into out, which it must fill: 0, or an error
int lzf_decompress_cpp(
    const std::uint8_t* in,
    std::size_t in_len,
    std::uint8_t* out,
    std::size_t out_len
) noexcept;

// A record as an LZF one decompresses to (the doubles start_t and delta_t, then an int32 intensity a datapoint),
// appended to out in the token format of the newer MSProfile.bin: start_t, delta_t, 0x90 and the datapoint count,
// minus the zeros first, then the tokens, without the zeros last. 0, or an error
int encode_record_cpp(
    const std::uint8_t* raw,
    std::size_t raw_len,
    std::vector<std::uint8_t>& out
);

// The records of MSProfile.bin into out: those of lzf[i] != 0 decompressed to uncompressed_counts[i] bytes and
// encoded, the others copied. Where each starts in out and its bytes into offsets_out and byte_counts_out.
// 0, or an error of the record *failed_record
int rewrite_lzf_records_cpp(
    const std::uint8_t* base,
    std::size_t base_len,
    const std::int64_t* offsets,
    const std::int64_t* byte_counts,
    const std::int64_t* uncompressed_counts,
    const std::uint8_t* lzf,
    std::size_t N_records,
    std::vector<std::uint8_t>& out,
    std::int64_t* offsets_out,
    std::int64_t* byte_counts_out,
    std::size_t* failed_record
) noexcept;

} // namespace inten
