#include "serve_kv.hpp"
#include "score_queries.hpp"
#include "normalize_scores.hpp"
#include "accumulate_values.hpp"

namespace {
using namespace brisk;

int validate_call(int batches, int sequences, int start, int stride,
                  int valid, int mask, int flags) {
    // Widen before addition: even a hostile signed-32 argument cannot overflow.
    const std::int64_t end = std::int64_t(start) + std::int64_t(sequences);
    if (batches < 1 || batches > max_batches || sequences < 1 ||
        sequences > max_sequence || start < 0 || stride < 1 ||
        stride > max_sequence || end > stride) return invalid_shape;
    if (valid != start) return invalid_history;
    if (mask < 0 || mask > 1 || (sequences > 1 && mask != 1)) return invalid_mask;
    if (flags != 0) return invalid_flags;
    return success;
}

void load_queries(const qo_pack_t* q0, const qo_pack_t* q1,
                  const qo_pack_t* q2, const qo_pack_t* q3,
                  query_stream rep0[lanes], query_stream rep1[lanes],
                  query_stream rep2[lanes], query_stream rep3[lanes],
                  int host_sequences, int tile_start, const TileRequest& request) {
#pragma HLS INLINE off
    for (int batch = 0; batch < request.batches; ++batch) {
        for (int head = 0; head < local_kv_heads; ++head) {
            for (int pair = 0; pair < query_pairs_per_head; ++pair) {
                for (int row = 0; row < request.sequences; ++row) {
                    for (int rep = 0; rep < representatives; ++rep) {
#pragma HLS PIPELINE II=2 style=flp
                        const std::size_t index =
                            (((std::size_t(batch) * host_sequences + tile_start + row)
                              * local_query_heads + head * representatives + rep)
                             * qo_words_per_head + pair * 2);
                        const QPair p0 = {{q0[index], q0[index + 1]}};
                        const QPair p1 = {{q1[index], q1[index + 1]}};
                        const QPair p2 = {{q2[index], q2[index + 1]}};
                        const QPair p3 = {{q3[index], q3[index + 1]}};
                        // Feed corresponding work to every lane before moving on.
                        // A lane-at-a-time loader can close a finite-FIFO cycle at
                        // the shared normalizer's four-lane join.
                        switch (rep) {
                            case 0: rep0[0].write(p0); rep0[1].write(p1);
                                    rep0[2].write(p2); rep0[3].write(p3); break;
                            case 1: rep1[0].write(p0); rep1[1].write(p1);
                                    rep1[2].write(p2); rep1[3].write(p3); break;
                            case 2: rep2[0].write(p0); rep2[1].write(p1);
                                    rep2[2].write(p2); rep2[3].write(p3); break;
                            default: rep3[0].write(p0); rep3[1].write(p1);
                                     rep3[2].write(p2); rep3[3].write(p3); break;
                        }
                    }
                }
            }
        }
    }
}

cache_line_t load_active_line(const kv_pair_pack_t* active, std::size_t index) {
#pragma HLS INLINE
    cache_line_t line;
    line.range(255, 0) = active[index];
    line.range(511, 256) = active[index + 1];
    return line;
}

void load_incoming(const kv_pair_pack_t* active, incoming_stream incoming[lanes],
                   int host_sequences, int tile_start, const TileRequest& request) {
#pragma HLS INLINE off
    for (int batch = 0; batch < request.batches; ++batch) {
        for (int row = 0; row < request.sequences; ++row) {
            for (int head = 0; head < local_kv_heads; ++head) {
                for (int line = 0; line < lines_per_head; ++line) {
                    // Interleave lanes within a record, rather than filling a
                    // whole lane's tile before any other owner can make progress.
                    for (int lane = 0; lane < lanes; ++lane) {
#pragma HLS PIPELINE II=2 style=flp
                        const int global_head = lane * local_kv_heads + head;
                        const std::size_t index =
                            (((std::size_t(batch) * host_sequences + tile_start + row)
                              * kv_heads + global_head) * chunks_per_head + line * 2);
                        const cache_line_t packet = load_active_line(active, index);
                        switch (lane) {
                            case 0: incoming[0].write(packet); break;
                            case 1: incoming[1].write(packet); break;
                            case 2: incoming[2].write(packet); break;
                            default: incoming[3].write(packet); break;
                        }
                    }
                }
            }
        }
    }
}

void store_output(output_stream& output, qo_pack_t* destination,
                  int host_sequences, int tile_start, const TileRequest& request) {
#pragma HLS INLINE off
    for (int batch = 0; batch < request.batches; ++batch) {
        for (int row = 0; row < request.sequences; ++row) {
            for (int head = 0; head < local_query_heads; ++head) {
                for (int word = 0; word < qo_words_per_head; ++word) {
#pragma HLS PIPELINE II=1 style=flp
                    const std::size_t index =
                        (((std::size_t(batch) * host_sequences + tile_start + row)
                          * local_query_heads + head) * qo_words_per_head + word);
                    destination[index] = output.read();
                }
            }
        }
    }
}

void run_tile(const qo_pack_t* q0, const qo_pack_t* q1,
              const qo_pack_t* q2, const qo_pack_t* q3,
              const kv_pair_pack_t* active,
              cache_line_t* cache0, cache_line_t* cache1,
              cache_line_t* cache2, cache_line_t* cache3,
              qo_pack_t* out0, qo_pack_t* out1,
              qo_pack_t* out2, qo_pack_t* out3,
              int host_sequences, int tile_start,
              const TileRequest& request0, const TileRequest& request1,
              const TileRequest& request2, const TileRequest& request3) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
    // Literal lane fields are visible in this region, so the public serve_kv
    // wrapper selects exactly one static specialization at each call site.
    const TileRequest lane0 = {0, request0.batches, request0.sequences,
        request0.context_start, request0.cache_stride, request0.cache_identity};
    const TileRequest lane1 = {1, request1.batches, request1.sequences,
        request1.context_start, request1.cache_stride, request1.cache_identity};
    const TileRequest lane2 = {2, request2.batches, request2.sequences,
        request2.context_start, request2.cache_stride, request2.cache_identity};
    const TileRequest lane3 = {3, request3.batches, request3.sequences,
        request3.context_start, request3.cache_stride, request3.cache_identity};
    query_stream rep0[lanes], rep1[lanes], rep2[lanes], rep3[lanes];
    incoming_stream incoming[lanes];
    key_stream keys[lanes];
    value_stream values[lanes];
    score_stream scores[lanes];
    scale_stream scales[lanes];
    denominator_stream denominators[lanes];
    output_stream output[lanes];
#pragma HLS ARRAY_PARTITION variable=rep0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=rep1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=rep2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=rep3 complete dim=1
#pragma HLS ARRAY_PARTITION variable=incoming complete dim=1
#pragma HLS ARRAY_PARTITION variable=keys complete dim=1
#pragma HLS ARRAY_PARTITION variable=values complete dim=1
#pragma HLS ARRAY_PARTITION variable=scores complete dim=1
#pragma HLS ARRAY_PARTITION variable=scales complete dim=1
#pragma HLS ARRAY_PARTITION variable=denominators complete dim=1
#pragma HLS ARRAY_PARTITION variable=output complete dim=1
#pragma HLS STREAM variable=rep0 depth=query_depth
#pragma HLS STREAM variable=rep1 depth=query_depth
#pragma HLS STREAM variable=rep2 depth=query_depth
#pragma HLS STREAM variable=rep3 depth=query_depth
#pragma HLS STREAM variable=incoming depth=incoming_depth
#pragma HLS STREAM variable=keys depth=key_depth
#pragma HLS STREAM variable=values depth=value_depth
#pragma HLS STREAM variable=scores depth=score_depth
#pragma HLS STREAM variable=scales depth=scale_depth
#pragma HLS STREAM variable=denominators depth=denominator_depth
#pragma HLS STREAM variable=output depth=output_depth

    load_queries(q0, q1, q2, q3, rep0, rep1, rep2, rep3,
                 host_sequences, tile_start, lane0);
    load_incoming(active, incoming, host_sequences, tile_start, lane0);
    serve_kv(cache0, incoming[0], keys[0], values[0], lane0);
    serve_kv(cache1, incoming[1], keys[1], values[1], lane1);
    serve_kv(cache2, incoming[2], keys[2], values[2], lane2);
    serve_kv(cache3, incoming[3], keys[3], values[3], lane3);
    score_queries(rep0[0], rep1[0], rep2[0], rep3[0], keys[0], scores[0], lane0);
    score_queries(rep0[1], rep1[1], rep2[1], rep3[1], keys[1], scores[1], lane1);
    score_queries(rep0[2], rep1[2], rep2[2], rep3[2], keys[2], scores[2], lane2);
    score_queries(rep0[3], rep1[3], rep2[3], rep3[3], keys[3], scores[3], lane3);
    normalize_scores(scores, scales, denominators, lane0);
    accumulate_values(values[0], scales[0], denominators[0], output[0], lane0);
    accumulate_values(values[1], scales[1], denominators[1], output[1], lane1);
    accumulate_values(values[2], scales[2], denominators[2], output[2], lane2);
    accumulate_values(values[3], scales[3], denominators[3], output[3], lane3);
    store_output(output[0], out0, host_sequences, tile_start, lane0);
    store_output(output[1], out1, host_sequences, tile_start, lane1);
    store_output(output[2], out2, host_sequences, tile_start, lane2);
    store_output(output[3], out3, host_sequences, tile_start, lane3);
}

}  // namespace

extern "C" void llama3_fast_attention(
    const qo_pack_t* q0, const qo_pack_t* q1,
    const qo_pack_t* q2, const qo_pack_t* q3,
    const kv_pair_pack_t* active_kv,
    cache_line_t* cache_kv0, cache_line_t* cache_kv1,
    cache_line_t* cache_kv2, cache_line_t* cache_kv3,
    qo_pack_t* out0, qo_pack_t* out1, qo_pack_t* out2, qo_pack_t* out3,
    int batch_size, int seqlen, int start_pos, int cache_stride,
    int cache_valid_len, int mask_mode, int flags, int cache_identity,
    int* status_out) {
    // Memory mappings live in the native HLS configuration; status remains below.
    // Adapter depths do not alter the validated hardware shape bounds.
    // Lane-zero backing uses native M_AXI bundle gmem_kv0.
    // Lane-one backing uses native M_AXI bundle gmem_kv1.
    // Lane-two backing uses native M_AXI bundle gmem_kv2.
    // Lane-three backing uses native M_AXI bundle gmem_kv3.
#pragma HLS INTERFACE m_axi port=status_out bundle=gmem_status depth=1
#pragma HLS INTERFACE s_axilite port=q0 bundle=control
#pragma HLS INTERFACE s_axilite port=q1 bundle=control
#pragma HLS INTERFACE s_axilite port=q2 bundle=control
#pragma HLS INTERFACE s_axilite port=q3 bundle=control
#pragma HLS INTERFACE s_axilite port=active_kv bundle=control
#pragma HLS INTERFACE s_axilite port=cache_kv0 bundle=control
#pragma HLS INTERFACE s_axilite port=cache_kv1 bundle=control
#pragma HLS INTERFACE s_axilite port=cache_kv2 bundle=control
#pragma HLS INTERFACE s_axilite port=cache_kv3 bundle=control
#pragma HLS INTERFACE s_axilite port=out0 bundle=control
#pragma HLS INTERFACE s_axilite port=out1 bundle=control
#pragma HLS INTERFACE s_axilite port=out2 bundle=control
#pragma HLS INTERFACE s_axilite port=out3 bundle=control
#pragma HLS INTERFACE s_axilite port=batch_size bundle=control
#pragma HLS INTERFACE s_axilite port=seqlen bundle=control
#pragma HLS INTERFACE s_axilite port=start_pos bundle=control
#pragma HLS INTERFACE s_axilite port=cache_stride bundle=control
#pragma HLS INTERFACE s_axilite port=cache_valid_len bundle=control
#pragma HLS INTERFACE s_axilite port=mask_mode bundle=control
#pragma HLS INTERFACE s_axilite port=flags bundle=control
#pragma HLS INTERFACE s_axilite port=cache_identity bundle=control
#pragma HLS INTERFACE s_axilite port=status_out bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
    const int status = validate_call(batch_size, seqlen, start_pos, cache_stride,
                                     cache_valid_len, mask_mode, flags);
    if (status != brisk::success) {
        status_out[0] = status;
        return;
    }
    for (int tile_start = 0; tile_start < seqlen; tile_start += brisk::tile_rows) {
        const int remaining = seqlen - tile_start;
        const int count = remaining < brisk::tile_rows ? remaining : brisk::tile_rows;
        const brisk::TileRequest request0 =
            {0, batch_size, count, start_pos + tile_start, cache_stride, cache_identity};
        const brisk::TileRequest request1 =
            {1, batch_size, count, start_pos + tile_start, cache_stride, cache_identity};
        const brisk::TileRequest request2 =
            {2, batch_size, count, start_pos + tile_start, cache_stride, cache_identity};
        const brisk::TileRequest request3 =
            {3, batch_size, count, start_pos + tile_start, cache_stride, cache_identity};
        run_tile(q0, q1, q2, q3, active_kv, cache_kv0, cache_kv1, cache_kv2,
                 cache_kv3, out0, out1, out2, out3, seqlen, tile_start,
                 request0, request1, request2, request3);
    }
    status_out[0] = brisk::success;
}
