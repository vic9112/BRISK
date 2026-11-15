#pragma once

#include "attention.hpp"
#include <hls_fence.h>

namespace brisk {
namespace detail {

struct ResidentTag {
    std::uint32_t identity;
    unsigned short position;
    bool batch;
    bool valid;
};

struct LayerContext {
    std::uint32_t identity;
    int batches;
    int stride;
    bool valid;
};

inline void emit_line(cache_line_t line, key_stream& keys,
                      value_stream& values) {
#pragma HLS INLINE
    for (int half = 0; half < chunks_per_line; ++half) {
#pragma HLS PIPELINE II=1 style=flp
        const kv_pair_pack_t pair = line.range(half * 256 + 255, half * 256);
        KChunk key;
        VChunk value;
        for (int dimension = 0; dimension < chunk_elements; ++dimension) {
#pragma HLS UNROLL
            key.values[dimension] = bf16_t(pair.range(dimension * 16 + 15,
                                                     dimension * 16));
            value.values[dimension] = bf16_t(pair.range(128 + dimension * 16 + 15,
                                                       128 + dimension * 16));
        }
        keys.write(key);
        values.write(value);
    }
}

// A template specialization is a physical ownership boundary, not a different
// algorithm: each lane has its own payload, tags, and layer contexts.
template<int Lane>
void serve_kv_lane(cache_line_t* backing, incoming_stream& incoming,
                   key_stream& keys, value_stream& values,
                   const TileRequest& request) {
#pragma HLS INLINE off
    static_assert(Lane >= 0 && Lane < lanes, "resident owner lane");
    static cache_line_t payload[resident_positions][lines_per_record];
    static ResidentTag tags[resident_positions] = {};
    static LayerContext contexts[layers] = {};
    // Five depth banks give 256 positions * 16 lines = 4096 rows per bank.
    // The 512-bit width is mapped by the tool; reports must verify the URAM use.
#pragma HLS ARRAY_PARTITION variable=payload block factor=5 dim=1
#pragma HLS BIND_STORAGE variable=payload type=ram_2p impl=uram
#pragma HLS BIND_STORAGE variable=tags type=ram_2p impl=bram
#pragma HLS AGGREGATE variable=tags compact=bit
#pragma HLS BIND_STORAGE variable=contexts type=ram_2p impl=bram

    const std::uint32_t identity = std::uint32_t(request.cache_identity);
    const int layer = int(identity & (layers - 1));
    const int layer_base = layer * positions_per_layer;
    const int window = positions_per_layer / request.batches;
    const LayerContext previous = contexts[layer];
    const bool new_context = request.context_start == 0 || !previous.valid ||
        previous.identity != identity || previous.batches != request.batches ||
        previous.stride != request.cache_stride;
    if (new_context) {
        // A fresh start retires only this layer. Retired data must never be
        // written through the new context's backing pointer, even at equal ID.
        for (int slot = 0; slot < positions_per_layer; ++slot) {
#pragma HLS PIPELINE II=1
            tags[layer_base + slot].valid = false;
        }
    }
    contexts[layer] = {identity, request.batches, request.cache_stride, true};

    for (int batch = 0; batch < request.batches; ++batch) {
        for (int position = 0;
             position < request.context_start + request.sequences; ++position) {
            const int slot = layer_base + batch * window + position % window;
            const ResidentTag tag = tags[slot];
            const bool hit = tag.valid && tag.identity == identity &&
                tag.batch == bool(batch) && tag.position == position;

            if (position < request.context_start) {
                // A historical miss is a direct, nonallocating read. It cannot
                // evict a later resident position that this call still needs.
                // Keep the memory sources in separate whole-record control
                // regions: an exact hit must not issue a backing read request.
                if (hit) {
                    for (int head = 0; head < local_kv_heads; ++head) {
                        for (int line = 0; line < lines_per_head; ++line) {
                            const cache_line_t selected =
                                payload[slot][head * lines_per_head + line];
                            emit_line(selected, keys, values);
                        }
                    }
                } else {
                    for (int head = 0; head < local_kv_heads; ++head) {
                        for (int line = 0; line < lines_per_head; ++line) {
                            const cache_line_t selected = backing[backing_index(
                                batch, position, head, line, request.cache_stride)];
                            emit_line(selected, keys, values);
                        }
                    }
                }
            } else {
                // Every valid tag belongs to an incoming, dirty live record.
                // Preserve its exact BF16 bytes before reusing the payload slot.
                if (tag.valid) {
                    for (int head = 0; head < local_kv_heads; ++head) {
                        for (int line = 0; line < lines_per_head; ++line) {
#pragma HLS PIPELINE II=1
                            backing[backing_index(int(tag.batch), tag.position,
                                head, line, request.cache_stride)] =
                                payload[slot][head * lines_per_head + line];
                        }
                    }
                    // This explicit ordering constraint separates the external
                    // writes from payload reuse and tag publication. AXI response
                    // completion must additionally be checked in generated RTL.
                    hls::fence(backing, payload, tags);
                }
                tags[slot].valid = false;
                for (int head = 0; head < local_kv_heads; ++head) {
                    for (int line = 0; line < lines_per_head; ++line) {
                        const cache_line_t selected = incoming.read();
                        payload[slot][head * lines_per_head + line] = selected;
                        emit_line(selected, keys, values);
                    }
                }
                // Publish only after both local heads' K and V have arrived.
                tags[slot] = {identity, static_cast<unsigned short>(position),
                              bool(batch), true};
            }
        }
    }
}

}  // namespace detail

inline void serve_kv(cache_line_t* backing, incoming_stream& incoming,
                     key_stream& keys, value_stream& values,
                     const TileRequest& request) {
#pragma HLS INLINE
    // Root call sites supply constant lanes, so the dispatch is elaboration-time.
    switch (request.lane) {
        case 0: detail::serve_kv_lane<0>(backing, incoming, keys, values, request); break;
        case 1: detail::serve_kv_lane<1>(backing, incoming, keys, values, request); break;
        case 2: detail::serve_kv_lane<2>(backing, incoming, keys, values, request); break;
        default: detail::serve_kv_lane<3>(backing, incoming, keys, values, request); break;
    }
}

}  // namespace brisk
