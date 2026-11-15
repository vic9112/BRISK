#pragma once

#include "attention.hpp"

namespace brisk {

inline void accumulate_values(value_stream& values, scale_stream& scales,
                              denominator_stream& denominators,
                              output_stream& output, const TileRequest& request) {
#pragma HLS INLINE off
    float weighted[tile_rows][local_kv_heads][representatives][dimensions];
#pragma HLS ARRAY_PARTITION variable=weighted complete dim=3
#pragma HLS ARRAY_PARTITION variable=weighted cyclic factor=8 dim=4
#pragma HLS BIND_STORAGE variable=weighted type=ram_t2p impl=bram
    float value[dimensions];
#pragma HLS ARRAY_PARTITION variable=value cyclic factor=8 dim=1

    for (int batch = 0; batch < request.batches; ++batch) {
        for (int position = 0;
             position < request.context_start + request.sequences; ++position) {
            for (int head = 0; head < local_kv_heads; ++head) {
                // The current value head is the only history value retained here.
                for (int chunk = 0; chunk < chunks_per_head; ++chunk) {
#pragma HLS PIPELINE II=1 style=flp
                    const VChunk packet = values.read();
                    for (int d = 0; d < chunk_elements; ++d) {
#pragma HLS UNROLL
                        value[chunk * chunk_elements + d] = bf16_to_float(packet.values[d]);
                    }
                }
                for (int row = first_visible_row(position, request.context_start);
                     row < request.sequences; ++row) {
                    const Scale scale = scales.read();
                    for (int chunk = 0; chunk < chunks_per_head; ++chunk) {
#pragma HLS PIPELINE II=1
                        for (int rep = 0; rep < representatives; ++rep) {
#pragma HLS UNROLL
                            for (int d = 0; d < chunk_elements; ++d) {
#pragma HLS UNROLL
                                const int dimension = chunk * chunk_elements + d;
                                const float v = value[dimension];
                                if (position == 0) {
                                    weighted[row][head][rep][dimension] = v;
                                } else {
                                    const float old = weighted[row][head][rep][dimension];
                                    const float rescaled = scale.max_grew[rep]
                                        ? old * scale.previous[rep] : old;
                                    weighted[row][head][rep][dimension] =
                                        rescaled + v * scale.current[rep];
                                }
                            }
                        }
                    }
                    if (position == request.context_start + row) {
                        const Denominator denominator = denominators.read();
                        for (int rep = 0; rep < representatives; ++rep) {
                            ap_uint<128> lower = 0;
                            for (int chunk = 0; chunk < chunks_per_head; ++chunk) {
#pragma HLS PIPELINE II=1 style=flp
                                // Constant slices form one half-word without a
                                // serial, dynamic 256-bit read/modify/write chain.
                                ap_uint<128> half = 0;
                                for (int d = 0; d < chunk_elements; ++d) {
#pragma HLS UNROLL
                                    const int dimension = chunk * chunk_elements + d;
                                    const float normalized = weighted[row][head][rep][dimension]
                                        / denominator.values[rep];
                                    half.range(d * 16 + 15, d * 16) = float_to_bf16(normalized);
                                }
                                if (chunk % 2 == 0) {
                                    lower = half;
                                } else {
                                    qo_pack_t word;
                                    word.range(127, 0) = lower;
                                    word.range(255, 128) = half;
                                    output.write(word);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

}  // namespace brisk
