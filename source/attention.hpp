#pragma once

#include "brisk_stream_types.hpp"
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace brisk {

// Model geometry and packed-word geometry, shared by all five functions.
constexpr int query_heads = 32;
constexpr int kv_heads = 8;
constexpr int lanes = 4;
constexpr int local_kv_heads = kv_heads / lanes;
constexpr int representatives = query_heads / kv_heads;
constexpr int local_query_heads = local_kv_heads * representatives;
constexpr int dimensions = 128;
constexpr int layers = 32;
constexpr int max_batches = 2;
constexpr int max_sequence = 8192;
constexpr int tile_rows = 16;
constexpr int chunk_elements = 8;
constexpr int chunks_per_head = dimensions / chunk_elements;
constexpr int qo_elements = 16;
constexpr int qo_words_per_head = dimensions / qo_elements;
constexpr int query_pairs_per_head = qo_words_per_head / 2;
constexpr int chunks_per_line = 2;
constexpr int lines_per_head = chunks_per_head / chunks_per_line;
constexpr int lines_per_record = local_kv_heads * lines_per_head;

// Forty positions per layer are shared between the batches, not between lanes.
constexpr int positions_per_layer = 40;
constexpr int resident_positions = layers * positions_per_layer;
constexpr float score_scale = 0.08838834764831845f;  // 1 / sqrt(128)

// Chosen channel capacities from the specification; these are not minimum proofs.
// They decouple AXI/URAM service, score latency, and the shared normalizer join.
constexpr int incoming_depth = 16;     // One complete two-head K/V record.
constexpr int query_depth = 8;         // Eight QPairs per lane and representative.
constexpr int key_depth = 4;           // Four eight-dimension key chunks.
constexpr int value_depth = 128;       // Eight complete value heads of slack.
constexpr int score_depth = 8;         // Eight four-representative score packets.
constexpr int scale_depth = 64;        // Decouples normalization from SV recurrence.
constexpr int denominator_depth = 32;  // One batch's 16 rows * two local heads.
constexpr int output_depth = 64;       // One row's eight heads * eight output words.

enum CallStatus {
    success = 0,
    invalid_shape = 1,
    invalid_history = 2,
    invalid_mask = 3,
    invalid_flags = 4
};

// Memory mappings and pointer depths are owned by the native HLS configuration.
// Adapter extents do not set the algorithm's supported hardware shape bounds.
// Each backing pointer selects one layer; there is no layer address term.
// A full lane backing extent is max_batches * max_sequence * local_kv_heads *
// lines_per_head cache lines, per supplied layer pointer.
// Adapter-only extent changes require byte-identical generated hardware.

static_assert(lanes * local_kv_heads == kv_heads, "lane head coverage");
static_assert(kv_heads * representatives == query_heads, "GQA coverage");
static_assert(positions_per_layer % max_batches == 0, "batch slot partition");
static_assert(sizeof(int) == 4, "the control ABI uses signed 32-bit integers");

inline float bf16_to_float(bf16_t value) {
#pragma HLS INLINE
    union { std::uint32_t bits; float number; } conversion;
    conversion.bits = std::uint32_t(value) << 16;
    return conversion.number;
}

inline bf16_t float_to_bf16(float value) {
#pragma HLS INLINE
    union { float number; std::uint32_t bits; } conversion;
    conversion.number = value;
    if ((conversion.bits & 0x7fffffffU) > 0x7f800000U)
        return bf16_t((conversion.bits >> 16) | 0x0040U);
    // Round to nearest, with a tie going to an even retained mantissa bit.
    conversion.bits += 0x7fffU + ((conversion.bits >> 16) & 1U);
    return bf16_t(conversion.bits >> 16);
}

inline float reduce_eight(const float values[chunk_elements]) {
#pragma HLS INLINE
    const float a = values[0] + values[1];
    const float b = values[2] + values[3];
    const float c = values[4] + values[5];
    const float d = values[6] + values[7];
    return (a + b) + (c + d);
}

inline std::size_t backing_index(int batch, int position, int head,
                                 int line, int stride) {
#pragma HLS INLINE
    return (((std::size_t(batch) * stride + position) * local_kv_heads + head)
            * lines_per_head + line);
}

inline int first_visible_row(int position, int context_start) {
#pragma HLS INLINE
    return position < context_start ? 0 : position - context_start;
}

}  // namespace brisk
