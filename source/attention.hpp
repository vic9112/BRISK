//********************************************************
//* Project         : FPGenius - FPT'26 Design Competition Track B
//* Filename        : attention.hpp
//* Author          : VicChen (KUAN-HSI, CHEN)
//* Created On      : 2026/07/10
//* Function        : Defines the BF16 types, constants, numeric helpers, and 22-argument kernel ABI.
//********************************************************

#pragma once

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ap_int.h>
#include <hls_stream.h>
using bf16_t = std::uint16_t;
using kv_pack_t = ap_uint<128>;
using kv_pair_pack_t = ap_uint<256>;
using qo_pack_t = ap_uint<256>;
using cache_line_t = ap_uint<512>;
using kv_credit_t = ap_uint<1>;
constexpr int FATTN_NUM_HEADS = 32;
constexpr int FATTN_NUM_KV_HEADS = 8;
constexpr int FATTN_NUM_LANES = 4;
constexpr int FATTN_KV_SLOTS_PER_LANE = 2;
constexpr int FATTN_DUAL_HEAD_CACHE_BANKS = FATTN_NUM_LANES;
constexpr int FATTN_DUAL_HEAD_CACHE_SLOTS = FATTN_KV_SLOTS_PER_LANE;
constexpr int FATTN_HEAD_DIM = 128;
constexpr int FATTN_MODEL_DIM = FATTN_NUM_HEADS * FATTN_HEAD_DIM;
constexpr int FATTN_KV_REP = 4;
constexpr int FATTN_HYBRID_FIP_REPS = FATTN_KV_REP / 2;
constexpr int FATTN_FIP_REP0 = 1;
constexpr int FATTN_FIP_REP1 = 2;
constexpr bool rep_uses_fip(int rep) {
  return rep == FATTN_FIP_REP0 || rep == FATTN_FIP_REP1;
}
constexpr int fip_rep_index(int rep) {
  return rep - FATTN_FIP_REP0;
}
constexpr int fip_rep_from_index(int index) {
  return FATTN_FIP_REP0 + index;
}
constexpr int classical_rep_index(int rep) {
  return rep == 0 ? 0 : rep - FATTN_HYBRID_FIP_REPS;
}
constexpr int classical_rep_from_index(int index) {
  return index == 0 ? 0 : FATTN_HYBRID_FIP_REPS + index;
}
constexpr int FATTN_Q_HEADS_PER_LANE =
    FATTN_KV_SLOTS_PER_LANE * FATTN_KV_REP;
constexpr int FATTN_MAX_SEQ_LEN = 8192;
constexpr int FATTN_MAX_BATCH = 2;
constexpr int FATTN_TILE_TOKENS = 16;
static_assert(FATTN_TILE_TOKENS > 0 && FATTN_TILE_TOKENS <= 16,
              "query-resident rows must fit the verified temporal state bound");
constexpr int FATTN_QO_PACK_ELEMS = 16;
constexpr int FATTN_QO_HEAD_PACKS = FATTN_HEAD_DIM / FATTN_QO_PACK_ELEMS;
constexpr int FATTN_KV_PACK_ELEMS = 8;
constexpr int FATTN_KV_HEAD_PACKS = FATTN_HEAD_DIM / FATTN_KV_PACK_ELEMS;
constexpr int FATTN_CACHE_LINE_PACKS = 2;
constexpr int FATTN_CACHE_LINES_PER_SLOT =
    FATTN_KV_HEAD_PACKS / FATTN_CACHE_LINE_PACKS;
constexpr int FATTN_ACTIVE_PLANE_WORDS = FATTN_KV_HEAD_PACKS / 2;
constexpr int FATTN_VALUE_LANES = 8;
constexpr int FATTN_VALUE_CHUNKS = FATTN_HEAD_DIM / FATTN_VALUE_LANES;
constexpr int FATTN_PLANE_LINES =
    FATTN_VALUE_CHUNKS / 4;
constexpr int FATTN_PLANE_CHUNKS =
    FATTN_VALUE_CHUNKS / FATTN_PLANE_LINES;
constexpr int FATTN_DUAL_PORT_ACTIVE_LINES_MAX =
    FATTN_MAX_BATCH * FATTN_TILE_TOKENS * FATTN_KV_SLOTS_PER_LANE *
    FATTN_PLANE_LINES;
#define FATTN_COMPACT_B2S8_CACHE_STRIDE 19
#define FATTN_QO_INTERFACE_DEPTH 1048576
#define FATTN_ACTIVE_DEPTH 524288
#define FATTN_CACHE_DEPTH 131072
static_assert(
    1048576 ==
        FATTN_MAX_BATCH * FATTN_MAX_SEQ_LEN * FATTN_Q_HEADS_PER_LANE *
            FATTN_QO_HEAD_PACKS,
    "q/out depth must cover the complete B2/S8192 boundary");
static_assert(
    524288 ==
        FATTN_MAX_BATCH * FATTN_MAX_SEQ_LEN * FATTN_KV_SLOTS_PER_LANE *
            FATTN_PLANE_LINES * FATTN_NUM_LANES,
    "split active K/V depth must cover the complete B2/S8192 boundary");
static_assert(
    131072 ==
        FATTN_MAX_BATCH * FATTN_MAX_SEQ_LEN * FATTN_KV_SLOTS_PER_LANE *
            FATTN_PLANE_LINES,
    "each split cache K/V depth must cover the complete B2/S8192 boundary");
constexpr int FATTN_DUAL_PORT_PREFETCH_LINES = 2;
constexpr int FATTN_ELASTIC_V_RESPONSE_LINES = 2;
constexpr int FATTN_ELASTIC_CREDIT_DEPTH = 2;
static_assert(FATTN_ELASTIC_CREDIT_DEPTH ==
                  FATTN_DUAL_PORT_PREFETCH_LINES,
              "registered K credit count must equal payload capacity");
static_assert(FATTN_ELASTIC_CREDIT_DEPTH ==
                  FATTN_ELASTIC_V_RESPONSE_LINES,
              "registered V credit count must equal payload capacity");
static_assert(FATTN_PLANE_LINES * 4 ==
                  FATTN_VALUE_CHUNKS,
              "dual-port plane lines must cover one complete head");
constexpr int FATTN_HALFBEAT_CHUNKS = FATTN_CACHE_LINE_PACKS;
constexpr int FATTN_HALFBEATS_PER_HEAD =
    FATTN_VALUE_CHUNKS / FATTN_HALFBEAT_CHUNKS;
constexpr int FATTN_V_CREDIT_REQUIRED_CHUNKS = 112;
constexpr int FATTN_V_CREDIT_CHUNKS = 128;
constexpr int FATTN_V_CREDIT_LINES =
    (FATTN_V_CREDIT_CHUNKS + FATTN_PLANE_CHUNKS - 1) /
    FATTN_PLANE_CHUNKS;
static_assert(
    FATTN_V_CREDIT_CHUNKS >= FATTN_V_CREDIT_REQUIRED_CHUNKS,
    "V credit window must cover the exact score/scale pipeline flight");
static_assert(
    FATTN_V_CREDIT_LINES * FATTN_PLANE_CHUNKS >=
        FATTN_V_CREDIT_REQUIRED_CHUNKS,
    "wide V credit lines must cover the exact score/scale flight");
constexpr int FATTN_KV_PACK_BITS = 128;
constexpr int FATTN_KV_PAIR_PACK_BITS = 256;
constexpr int FATTN_QO_PACK_BITS = 256;
constexpr int FATTN_CACHE_LINE_BITS = 512;
constexpr float FATTN_INV_SQRT_HEAD_DIM = 0.08838834764831845f;
constexpr int FATTN_FLAG_ROUND_QK_BF16 = 1 << 1;
constexpr int FATTN_FLAG_CACHE_PREUPDATED = 1 << 2;
constexpr int FATTN_VALID_FLAGS =
    FATTN_FLAG_ROUND_QK_BF16 | FATTN_FLAG_CACHE_PREUPDATED;
constexpr int FATTN_MASK_NONE = 0;
constexpr int FATTN_MASK_CAUSAL = 1;
constexpr int FATTN_STATUS_OK = 0;
constexpr int FATTN_STATUS_INVALID_SHAPE = 1;
constexpr int FATTN_STATUS_INVALID_CACHE_STATE = 2;
constexpr int FATTN_STATUS_INVALID_MASK = 3;
constexpr int FATTN_STATUS_INVALID_FLAGS = 4;
constexpr int FATTN_STATUS_CACHE_BUSY = 5;
inline int validate_boundary_flags(int flags) {
#pragma HLS INLINE
  if ((flags & ~FATTN_VALID_FLAGS) != 0) return FATTN_STATUS_INVALID_FLAGS;
  return FATTN_STATUS_OK;
}
struct TilePlan {
  int batches;
  int sequences;
  int context_start;
  int cache_stride;
  int cache_actions;
  int visible_actions;
  int qk_actions;
  int halfbeat_actions;
  int softmax_actions;
  int normalize_actions;
  int normalize_halfbeat_actions;
};
#define FATTN_BOUND_WORKLOAD 0
#define FATTN_BOUND_BATCH_SIZE 0
#define FATTN_BOUND_SEQLEN 0
#define FATTN_BOUND_START_POS 0
#define FATTN_BOUND_CACHE_STRIDE 0
#define FATTN_BOUND_CACHE_VALID_LEN 0
#define FATTN_BOUND_MASK_MODE 0
#define FATTN_BOUND_FLAGS 0
static_assert(FATTN_NUM_LANES * FATTN_KV_SLOTS_PER_LANE ==
                  FATTN_NUM_KV_HEADS,
              "four lanes must cover every KV head");
static_assert(FATTN_NUM_KV_HEADS * FATTN_KV_REP == FATTN_NUM_HEADS,
              "GQA replication must cover every query head");
static_assert(sizeof(qo_pack_t) * 8 == FATTN_QO_PACK_BITS,
              "Q/O word width changed");
static_assert(sizeof(kv_pair_pack_t) * 8 == FATTN_KV_PAIR_PACK_BITS,
              "active K/V word width changed");
static_assert(sizeof(cache_line_t) * 8 == FATTN_CACHE_LINE_BITS,
              "cache line width changed");
inline bf16_t qo_pack_get(qo_pack_t word, int lane) {
  return static_cast<bf16_t>(word.range(lane * 16 + 15, lane * 16));
}
template <int Lane>
inline bf16_t qo_pack_get_static(qo_pack_t word) {
  static_assert(Lane >= 0 && Lane < FATTN_QO_PACK_ELEMS,
                "Q/O lane is out of range");
  return static_cast<bf16_t>(word.range(Lane * 16 + 15, Lane * 16));
}
inline qo_pack_t qo_pack_set(qo_pack_t word, int lane,
                                         bf16_t value) {
  word.range(lane * 16 + 15, lane * 16) = value;
  return word;
}
inline bf16_t kv_pair_pack_get_k(kv_pair_pack_t word,
                                             int lane) {
  return static_cast<bf16_t>(word.range(lane * 16 + 15, lane * 16));
}
inline bf16_t kv_pair_pack_get_v(kv_pair_pack_t word,
                                             int lane) {
  const int shift = FATTN_KV_PACK_BITS + lane * 16;
  return static_cast<bf16_t>(word.range(shift + 15, shift));
}
template <int Lane>
inline bf16_t kv_pair_pack_get_k_static(
    kv_pair_pack_t word) {
  static_assert(Lane >= 0 && Lane < FATTN_KV_PACK_ELEMS,
                "K lane is out of range");
  return static_cast<bf16_t>(word.range(Lane * 16 + 15, Lane * 16));
}
template <int Lane>
inline bf16_t kv_pair_pack_get_v_static(
    kv_pair_pack_t word) {
  static_assert(Lane >= 0 && Lane < FATTN_KV_PACK_ELEMS,
                "V lane is out of range");
  constexpr int shift = FATTN_KV_PACK_BITS + Lane * 16;
  return static_cast<bf16_t>(word.range(shift + 15, shift));
}
inline kv_pair_pack_t kv_pair_pack_set_k(
    kv_pair_pack_t word, int lane, bf16_t value) {
  word.range(lane * 16 + 15, lane * 16) = value;
  return word;
}
inline kv_pair_pack_t kv_pair_pack_set_v(
    kv_pair_pack_t word, int lane, bf16_t value) {
  const int shift = FATTN_KV_PACK_BITS + lane * 16;
  word.range(shift + 15, shift) = value;
  return word;
}
inline cache_line_t make_cache_line(
    kv_pair_pack_t lower, kv_pair_pack_t upper) {
  cache_line_t line = 0;
  line.range(FATTN_KV_PAIR_PACK_BITS - 1, 0) = lower;
  line.range(FATTN_CACHE_LINE_BITS - 1, FATTN_KV_PAIR_PACK_BITS) = upper;
  return line;
}
inline kv_pair_pack_t cache_line_get(cache_line_t line,
                                                  int pack) {
  return static_cast<kv_pair_pack_t>(
      line >> (pack * FATTN_KV_PAIR_PACK_BITS));
}
inline kv_pack_t kv_pair_pack_get_k_pack(
    kv_pair_pack_t word) {
  return word.range(FATTN_KV_PACK_BITS - 1, 0);
}
inline kv_pack_t kv_pair_pack_get_v_pack(
    kv_pair_pack_t word) {
  return word.range(
      FATTN_KV_PAIR_PACK_BITS - 1, FATTN_KV_PACK_BITS);
}
inline kv_pair_pack_t make_kv_pair_pack(
    kv_pack_t key, kv_pack_t value) {
  kv_pair_pack_t word = 0;
  word.range(FATTN_KV_PACK_BITS - 1, 0) = key;
  word.range(FATTN_KV_PAIR_PACK_BITS - 1, FATTN_KV_PACK_BITS) = value;
  return word;
}
inline cache_line_t make_dimension_plane_line(
    kv_pack_t pack0, kv_pack_t pack1,
    kv_pack_t pack2, kv_pack_t pack3) {
  cache_line_t line = 0;
  line.range(127, 0) = pack0;
  line.range(255, 128) = pack1;
  line.range(383, 256) = pack2;
  line.range(511, 384) = pack3;
  return line;
}
inline kv_pack_t dim_plane_pack(
    cache_line_t line, int pack) {
  return static_cast<kv_pack_t>(
      line >> (pack * FATTN_KV_PACK_BITS));
}
inline float bf16_to_float(bf16_t value) {
  union {
    std::uint32_t bits;
    float value;
  } conversion;
  conversion.bits = std::uint32_t(value) << 16;
  return conversion.value;
}
inline bf16_t float_to_bf16(float value) {
  union {
    float value;
    std::uint32_t bits;
  } conversion;
  conversion.value = value;
  const std::uint32_t magnitude = conversion.bits & 0x7fffffffU;
  if (magnitude > 0x7f800000U) {
    return static_cast<bf16_t>((conversion.bits >> 16) | 0x0040U);
  }
  const std::uint32_t lsb = (conversion.bits >> 16) & 1U;
  conversion.bits += 0x7fffU + lsb;
  return static_cast<bf16_t>(conversion.bits >> 16);
}
inline float maybe_round_bf16(float value, bool enabled) {
#pragma HLS INLINE
  return enabled ? bf16_to_float(float_to_bf16(value)) : value;
}
inline float exp_softmax(float value) {
#pragma HLS INLINE
  return std::exp(value);
}
inline TilePlan make_tile_plan(int batches, int sequences,
                                          int context_start,
                                          int cache_stride) {
#pragma HLS INLINE
  const int tasks = batches * sequences * FATTN_KV_SLOTS_PER_LANE;
  const int visible_tokens =
      batches * FATTN_KV_SLOTS_PER_LANE *
      (sequences * context_start + sequences * (sequences + 1) / 2);
  TilePlan plan;
  plan.batches = batches;
  plan.sequences = sequences;
  plan.context_start = context_start;
  plan.cache_stride = cache_stride;
  plan.cache_actions = tasks * FATTN_CACHE_LINES_PER_SLOT;
  plan.visible_actions = visible_tokens * FATTN_VALUE_CHUNKS;
  plan.qk_actions = plan.visible_actions;
  plan.halfbeat_actions = visible_tokens * FATTN_HALFBEATS_PER_HEAD;
  plan.softmax_actions = visible_tokens + tasks;
  plan.normalize_actions =
      tasks * (FATTN_VALUE_CHUNKS + FATTN_KV_REP * FATTN_QO_HEAD_PACKS);
  plan.normalize_halfbeat_actions =
      tasks * (FATTN_HALFBEATS_PER_HEAD +
               FATTN_KV_REP * FATTN_QO_HEAD_PACKS);
  return plan;
}
inline kv_pack_t pack_bf16x8(
    const bf16_t lanes[FATTN_VALUE_LANES]) {
#pragma HLS INLINE
  kv_pack_t word = 0;
  for (int lane = 0; lane < FATTN_VALUE_LANES; ++lane) {
#pragma HLS UNROLL
    word.range(lane * 16 + 15, lane * 16) = lanes[lane];
  }
  return word;
}
inline qo_pack_t pack_output(kv_pack_t lower,
                                         kv_pack_t upper) {
#pragma HLS INLINE
  qo_pack_t word = 0;
  word.range(127, 0) = lower;
  word.range(255, 128) = upper;
  return word;
}
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
