#pragma once

#include <cstdint>
#include <ap_int.h>
#include <hls_stream.h>

using bf16_t = std::uint16_t;
using qo_pack_t = ap_uint<256>;
using kv_pair_pack_t = ap_uint<256>;
using cache_line_t = ap_uint<512>;

namespace brisk {

struct TileRequest {
    int lane;
    int batches;
    int sequences;
    int context_start;
    int cache_stride;
    int cache_identity;
};

struct QPair {
    qo_pack_t words[2];
};

struct KChunk {
    bf16_t values[8];
};

struct VChunk {
    bf16_t values[8];
};

struct Score {
    float values[4];
};

struct Scale {
    float previous[4];
    float current[4];
    bool max_grew[4];
};

struct Denominator {
    float values[4];
};

void serve_kv(cache_line_t *backing,
              hls::stream<cache_line_t>& incoming,
              hls::stream<KChunk>& keys,
              hls::stream<VChunk>& values,
              const TileRequest& request);

void score_queries(hls::stream<QPair>& q0,
                   hls::stream<QPair>& q1,
                   hls::stream<QPair>& q2,
                   hls::stream<QPair>& q3,
                   hls::stream<KChunk>& keys,
                   hls::stream<Score>& scores,
                   const TileRequest& request);

void normalize_scores(hls::stream<Score> scores[4],
                      hls::stream<Scale> scales[4],
                      hls::stream<Denominator> denominators[4],
                      const TileRequest& request);

void accumulate_values(hls::stream<VChunk>& values,
                       hls::stream<Scale>& scales,
                       hls::stream<Denominator>& denominators,
                       hls::stream<qo_pack_t>& output,
                       const TileRequest& request);

}  // namespace brisk

extern "C" void llama3_fast_attention(
    const qo_pack_t *q0, const qo_pack_t *q1,
    const qo_pack_t *q2, const qo_pack_t *q3,
    const kv_pair_pack_t *active_kv,
    cache_line_t *cache_kv0, cache_line_t *cache_kv1,
    cache_line_t *cache_kv2, cache_line_t *cache_kv3,
    qo_pack_t *out0, qo_pack_t *out1, qo_pack_t *out2,
    qo_pack_t *out3, int batch_size, int seqlen, int start_pos,
    int cache_stride, int cache_valid_len, int mask_mode, int flags,
    int cache_identity, int *status_out);
