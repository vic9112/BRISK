#pragma once

#include "attention.hpp"

namespace brisk {

inline void score_queries(query_stream& q0, query_stream& q1,
                          query_stream& q2, query_stream& q3,
                          key_stream& keys, score_stream& scores,
                          const TileRequest& request) {
#pragma HLS INLINE off
    // Only one batch's query tile is live. Representatives have independent
    // banks; two ports accept both words of a QPair without a wide FIFO.
    qo_pack_t query[local_kv_heads][tile_rows][representatives][qo_words_per_head];
#pragma HLS ARRAY_PARTITION variable=query complete dim=3
#pragma HLS BIND_STORAGE variable=query type=ram_t2p impl=bram
    float key[dimensions];
#pragma HLS ARRAY_PARTITION variable=key cyclic factor=8 dim=1

    for (int batch = 0; batch < request.batches; ++batch) {
        for (int head = 0; head < local_kv_heads; ++head) {
            for (int pair = 0; pair < query_pairs_per_head; ++pair) {
                for (int row = 0; row < request.sequences; ++row) {
#pragma HLS PIPELINE II=1 style=flp
                    const QPair packet0 = q0.read();
                    const QPair packet1 = q1.read();
                    const QPair packet2 = q2.read();
                    const QPair packet3 = q3.read();
                    for (int word = 0; word < 2; ++word) {
#pragma HLS UNROLL
                        query[head][row][0][pair * 2 + word] = packet0.words[word];
                        query[head][row][1][pair * 2 + word] = packet1.words[word];
                        query[head][row][2][pair * 2 + word] = packet2.words[word];
                        query[head][row][3][pair * 2 + word] = packet3.words[word];
                    }
                }
            }
        }
        for (int position = 0;
             position < request.context_start + request.sequences; ++position) {
            for (int head = 0; head < local_kv_heads; ++head) {
                // Read each selected key once, then reuse it over causal rows.
                for (int chunk = 0; chunk < chunks_per_head; ++chunk) {
#pragma HLS PIPELINE II=1 style=flp
                    const KChunk packet = keys.read();
                    for (int d = 0; d < chunk_elements; ++d) {
#pragma HLS UNROLL
                        key[chunk * chunk_elements + d] = bf16_to_float(packet.values[d]);
                    }
                }
                for (int row = first_visible_row(position, request.context_start);
                     row < request.sequences; ++row) {
                    float partial[representatives][chunks_per_head];
#pragma HLS ARRAY_PARTITION variable=partial complete dim=0
                    // The logical work item is eight dimensions for four GQA
                    // representatives. No score or probability is BF16-rounded.
                    for (int chunk = 0; chunk < chunks_per_head; ++chunk) {
#pragma HLS PIPELINE II=1
                        for (int rep = 0; rep < representatives; ++rep) {
#pragma HLS UNROLL
                            const qo_pack_t word = query[head][row][rep][chunk / 2];
                            float products[chunk_elements];
#pragma HLS ARRAY_PARTITION variable=products complete dim=1
                            for (int d = 0; d < chunk_elements; ++d) {
#pragma HLS UNROLL
                                const int offset = ((chunk % 2) * chunk_elements + d) * 16;
                                const bf16_t q = bf16_t(word.range(offset + 15, offset));
                                products[d] = bf16_to_float(q) * key[chunk * chunk_elements + d];
                            }
                            partial[rep][chunk] = reduce_eight(products);
                        }
                    }
                    Score packet;
                    for (int rep = 0; rep < representatives; ++rep) {
#pragma HLS UNROLL
                        float blocks[chunks_per_head / 2];
#pragma HLS ARRAY_PARTITION variable=blocks complete dim=1
                        for (int block = 0; block < chunks_per_head / 2; ++block) {
#pragma HLS UNROLL
                            blocks[block] = partial[rep][2 * block] + partial[rep][2 * block + 1];
                        }
                        packet.values[rep] = reduce_eight(blocks) * score_scale;
                    }
                    scores.write(packet);
                }
            }
        }
    }
}

}  // namespace brisk
