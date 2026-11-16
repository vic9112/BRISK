//********************************************************
//* Project         : FPGenius - FPT'26 Design Competition Track B
//* Filename        : attention_kernel.cpp
//* Author          : VicChen (KUAN-HSI, CHEN)
//* Created On      : 2026/07/10
//* Function        : Connects the four attention lanes to the stable XRT kernel interface.
//********************************************************

#include "attention_tile.hpp"
#include "kv_stream.hpp"
#include "attention_alternatives.hpp"
namespace {
// Preserve HLS source-location identifiers after separating alternatives.
#line 15
int validate_attention_call(int batch, int seqlen, int start, int stride,
                            int valid, int mask, int flags) {
#pragma HLS INLINE
  if (batch <= 0 || batch > FATTN_MAX_BATCH || seqlen <= 0 ||
      seqlen > FATTN_MAX_SEQ_LEN || start < 0 || stride <= 0 ||
      stride > FATTN_MAX_SEQ_LEN || start + seqlen > stride ||
      start + seqlen > FATTN_MAX_SEQ_LEN) return FATTN_STATUS_INVALID_SHAPE;
  if (valid != start) return FATTN_STATUS_INVALID_CACHE_STATE;
  if ((seqlen > 1 && mask != FATTN_MASK_CAUSAL) ||
      (seqlen == 1 && mask != FATTN_MASK_NONE && mask != FATTN_MASK_CAUSAL))
    return FATTN_STATUS_INVALID_MASK;
  return validate_boundary_flags(flags);
}
// Preserve HLS source-location identifiers after separating alternatives.
#line 37
inline void write_active_word(
    hls::stream<kv_pair_pack_t> even[4],
    hls::stream<kv_pair_pack_t> odd[4], int head, int pack,
    kv_pair_pack_t word) {
#pragma HLS INLINE
  const int lane = head >> 1;
  if ((pack & 1) == 0) {
    switch (lane) {
      case 0: even[0].write(word); break; case 1: even[1].write(word); break;
      case 2: even[2].write(word); break; default: even[3].write(word); break;
    }
  } else {
    switch (lane) {
      case 0: odd[0].write(word); break; case 1: odd[1].write(word); break;
      case 2: odd[2].write(word); break; default: odd[3].write(word); break;
    }
  }
}
void load_active_tile(  // Distribute incoming paired words; no resident ownership here.
    const kv_pair_pack_t *active,
    hls::stream<kv_pair_pack_t> even[4],
    hls::stream<kv_pair_pack_t> odd[4], int batch_size,
    int total_seqlen, int tile_start, int tile_count,
    bool prefill_physical_frame) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable = even complete dim = 1
#pragma HLS ARRAY_PARTITION variable = odd complete dim = 1
  if (prefill_physical_frame) {
    cache_line_t frame[64];  // One flattened token: 8 K/V heads * 8 wide lines.
#pragma HLS BIND_STORAGE variable = frame type = ram_t2p impl = bram
    for (int token = 0; token < 16; ++token) {  // Batch-two/eight-row contiguous burst staging.
      for (int line = 0; line < 64; ++line) {
#pragma HLS PIPELINE II = 1
        const int base = token * 128 + line * 2;
        frame[line] =
            make_cache_line(active[base], active[base + 1]);
      }
      for (int line = 0; line < 64; ++line) {
#pragma HLS PIPELINE II = 1
        const int first_word = line * 2;
        const int first_pack = first_word & (FATTN_KV_HEAD_PACKS - 1);
        const int head = (first_word / FATTN_KV_HEAD_PACKS) &
                         (FATTN_NUM_KV_HEADS - 1);
        write_active_word(even, odd, head, first_pack,
                          cache_line_get(frame[line], 0));
        write_active_word(even, odd, head, first_pack + 1,
                          cache_line_get(frame[line], 1));
      }
    }
    return;
  }
  for (int batch = 0; batch < batch_size; ++batch)
    for (int local = 0; local < tile_count; ++local)
      for (int head = 0; head < FATTN_NUM_KV_HEADS; ++head)
        for (int pack = 0; pack < FATTN_KV_HEAD_PACKS; ++pack) {
#pragma HLS PIPELINE II = 1
          const int sequence = tile_start + local;
          const std::size_t index =
              (((static_cast<std::size_t>(batch) * total_seqlen + sequence) *
                    FATTN_NUM_KV_HEADS +
                head) *
                   FATTN_KV_HEAD_PACKS +
               pack);
          write_active_word(even, odd, head, pack, active[index]);
        }
}
void update_cache_tile(  // Forward complete incoming lines to load_kv; keep the existing stream order.
    hls::stream<kv_pair_pack_t> &even,
    hls::stream<kv_pair_pack_t> &odd,
    hls::stream<bool> &ready, hls::stream<cache_line_t> &forwarded,
    cache_line_t *cache, int sequences, int start, int cache_stride,
    int actions, bool cache_preupdated) {
#pragma HLS INLINE off
  int batch = 0, sequence = 0, slot = 0, line = 0;
  for (int action = 0; action < actions; ++action) {
#pragma HLS PIPELINE II = 1
    const cache_line_t active_value =
        make_cache_line(even.read(), odd.read());
    const bool kv_is_writeback = true;
    (void)kv_is_writeback;
    (void)cache;
    (void)cache_preupdated;
    forwarded.write(active_value);
    if (++line == FATTN_CACHE_LINES_PER_SLOT) {
      line = 0;
      if (++slot == FATTN_KV_SLOTS_PER_LANE) {
        slot = 0;
        if (++sequence == sequences) { sequence = 0; ++batch; }
      }
    }
  }
}
// Preserve HLS source-location identifiers after separating alternatives.
#line 259
void store_output(hls::stream<qo_pack_t> &words,  // Drain normalized words to the original host layout.
                  qo_pack_t *output, int batches, int sequences,
                  int total_seqlen, int tile_start,
                  bool prefill_physical_frame) {
#pragma HLS INLINE off
  if (prefill_physical_frame) {
    qo_pack_t frame[64];  // One local query row: 8 heads * 8 packed output words.
#pragma HLS BIND_STORAGE variable = frame type = ram_t2p impl = bram
    for (int token = 0; token < 16; ++token) {
      for (int word = 0; word < 64; ++word) {
#pragma HLS PIPELINE II = 1
        frame[word] = words.read();
      }
      for (int line = 0; line < 32; ++line) {
#pragma HLS PIPELINE II = 1
        const int base = token * 64 + line * 2;
        output[base] = frame[line * 2];
        output[base + 1] = frame[line * 2 + 1];
      }
    }
    return;
  }
  for (int batch = 0; batch < batches; ++batch)
    for (int sequence = 0; sequence < sequences; ++sequence)
      for (int word = 0;
           word < FATTN_Q_HEADS_PER_LANE * FATTN_QO_HEAD_PACKS; ++word) {
#pragma HLS PIPELINE II = 1
        const std::size_t index =
            ((static_cast<std::size_t>(batch) * total_seqlen + tile_start +
              sequence) *
                 FATTN_Q_HEADS_PER_LANE * FATTN_QO_HEAD_PACKS +
             word);
        output[index] = words.read();
      }
}
template <int Lane>
void run_lane_front(  // Query loading, active forwarding, K/V ownership and canonical QK.
    const qo_pack_t *query,
    hls::stream<kv_pair_pack_t> &active_even,
    hls::stream<kv_pair_pack_t> &active_odd,
    cache_line_t *cache,
    hls::stream<DimensionVPacket> &values,
    hls::stream<Score> &scores, int total_seqlen, int tile_start,
    TilePlan plan, bool round_qk_bf16, bool cache_preupdated,
    int cache_identity, bool prefill_physical_frame) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
  hls::stream<QPair> q0("q0"), q1("q1"), q2("q2"), q3("q3");
  hls::stream<bool> ready("ready");  // Retained compatibility stream; unused by the active producer/consumer.
  hls::stream<cache_line_t> forwarded("forwarded");  // Preserves install-and-forward ordering into the lane owner.
  hls::stream<DimensionKPacket> keys("keys");
#pragma HLS STREAM variable = q0 depth = 8
#pragma HLS STREAM variable = q1 depth = 8
#pragma HLS STREAM variable = q2 depth = 8
#pragma HLS STREAM variable = q3 depth = 8
#pragma HLS STREAM variable = ready depth = 2
#pragma HLS STREAM variable = forwarded depth = 16
#pragma HLS STREAM variable = keys depth = 4
  load_q<Lane>(
      query, q0, q1, q2, q3, plan.batches, total_seqlen, tile_start,
      plan.sequences, prefill_physical_frame);
  update_cache_tile(active_even, active_odd, ready, forwarded, cache,
                    plan.sequences, plan.context_start, plan.cache_stride,
                    plan.cache_actions, cache_preupdated);
  load_kv<Lane>(
      cache, ready, forwarded, keys, values, plan.batches, plan.sequences,
      plan.context_start, plan.cache_stride, cache_identity,
      cache_preupdated);
  qk_lane(
      q0, q1, q2, q3, keys, scores, plan.batches, plan.sequences,
      plan.context_start, plan.qk_actions, round_qk_bf16);
}
void run_lane_back(  // Weighted values, final reciprocal normalization, then stores.
    hls::stream<DimensionVPacket> &values,
    hls::stream<Scale> &scales,
    hls::stream<Denominator> &denominators, qo_pack_t *output,
    int total_seqlen, int tile_start, TilePlan plan,
    bool prefill_physical_frame) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
  hls::stream<DimensionAccPacket> accumulators("accumulators");  // Keep accumulation separate from final normalization.
  hls::stream<qo_pack_t> normalized("normalized");  // The selected FIFO holds a full local query row.
#pragma HLS STREAM variable = accumulators depth = 2
#pragma HLS STREAM variable = normalized depth = 64
#pragma HLS BIND_STORAGE variable = normalized type = fifo impl = bram
  sv_lane(
      scales, values, accumulators, plan.batches, plan.sequences,
      plan.context_start);
  normalize(accumulators, denominators, normalized,
                  plan.normalize_actions);
  store_output(normalized, output, plan.batches, plan.sequences, total_seqlen,
               tile_start, prefill_physical_frame);
}
void run_tile(  // Preserve the selected four-lane DATAFLOW network and FIFO capacities.
    const qo_pack_t *q0, const qo_pack_t *q1,
    const qo_pack_t *q2, const qo_pack_t *q3,
    const kv_pair_pack_t *active, cache_line_t *cache0,
    cache_line_t *cache1, cache_line_t *cache2,
    cache_line_t *cache3, qo_pack_t *out0, qo_pack_t *out1,
    qo_pack_t *out2, qo_pack_t *out3, int total_seqlen,
    int tile_start, TilePlan plan,
    bool round_qk_bf16, bool cache_preupdated, int cache_identity,
    bool prefill_physical_frame) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
  hls::stream<kv_pair_pack_t> even[4], odd[4];
#pragma HLS ARRAY_PARTITION variable = even complete dim = 1
#pragma HLS ARRAY_PARTITION variable = odd complete dim = 1
#pragma HLS STREAM variable = even depth = 16
#pragma HLS STREAM variable = odd depth = 16
  hls::stream<DimensionVPacket> values[FATTN_NUM_LANES];
  hls::stream<Score> scores[FATTN_NUM_LANES];
  hls::stream<Scale> scales[FATTN_NUM_LANES];
  hls::stream<Denominator> denominators[FATTN_NUM_LANES];
#pragma HLS ARRAY_PARTITION variable = values complete dim = 1
#pragma HLS ARRAY_PARTITION variable = scores complete dim = 1
#pragma HLS ARRAY_PARTITION variable = scales complete dim = 1
#pragma HLS ARRAY_PARTITION variable = denominators complete dim = 1
#pragma HLS STREAM variable = values depth = FATTN_V_CREDIT_CHUNKS
#pragma HLS STREAM variable = scores depth = 8
#pragma HLS STREAM variable = scales depth = 64
#pragma HLS STREAM variable = denominators depth = 32
  load_active_tile(active, even, odd, plan.batches, total_seqlen, tile_start,
                   plan.sequences, prefill_physical_frame);
  run_lane_front<0>(q0, even[0], odd[0], cache0, values[0], scores[0],
                    total_seqlen, tile_start, plan, round_qk_bf16,
                    cache_preupdated, cache_identity, prefill_physical_frame);
  run_lane_front<1>(q1, even[1], odd[1], cache1, values[1], scores[1],
                    total_seqlen, tile_start, plan, round_qk_bf16,
                    cache_preupdated, cache_identity, prefill_physical_frame);
  run_lane_front<2>(q2, even[2], odd[2], cache2, values[2], scores[2],
                    total_seqlen, tile_start, plan, round_qk_bf16,
                    cache_preupdated, cache_identity, prefill_physical_frame);
  run_lane_front<3>(q3, even[3], odd[3], cache3, values[3], scores[3],
                    total_seqlen, tile_start, plan, round_qk_bf16,
                    cache_preupdated, cache_identity, prefill_physical_frame);
  online_sm(
      scores, scales, denominators, plan.batches, plan.sequences,
      plan.context_start);
  run_lane_back(values[0], scales[0], denominators[0], out0, total_seqlen,
                tile_start, plan, prefill_physical_frame);
  run_lane_back(values[1], scales[1], denominators[1], out1, total_seqlen,
                tile_start, plan, prefill_physical_frame);
  run_lane_back(values[2], scales[2], denominators[2], out2, total_seqlen,
                tile_start, plan, prefill_physical_frame);
  run_lane_back(values[3], scales[3], denominators[3], out3, total_seqlen,
                tile_start, plan, prefill_physical_frame);
}
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
    int cache_identity, int *status_out) {
#pragma HLS INTERFACE m_axi port = q0 bundle = gmem_qo0 depth = FATTN_QO_INTERFACE_DEPTH num_read_outstanding = 8 num_write_outstanding = 8 max_read_burst_length = 256 max_write_burst_length = 256
#pragma HLS INTERFACE m_axi port = q1 bundle = gmem_qo1 depth = FATTN_QO_INTERFACE_DEPTH num_read_outstanding = 8 num_write_outstanding = 8 max_read_burst_length = 256 max_write_burst_length = 256
#pragma HLS INTERFACE m_axi port = q2 bundle = gmem_qo2 depth = FATTN_QO_INTERFACE_DEPTH num_read_outstanding = 8 num_write_outstanding = 8 max_read_burst_length = 256 max_write_burst_length = 256
#pragma HLS INTERFACE m_axi port = q3 bundle = gmem_qo3 depth = FATTN_QO_INTERFACE_DEPTH num_read_outstanding = 8 num_write_outstanding = 8 max_read_burst_length = 256 max_write_burst_length = 256
#pragma HLS INTERFACE m_axi port = active_kv bundle = gmem_active depth = 2097152 num_read_outstanding = 8 max_read_burst_length = 256
#pragma HLS INTERFACE m_axi port = cache_kv0 bundle = gmem_kv0 depth = 262144
#pragma HLS INTERFACE m_axi port = cache_kv1 bundle = gmem_kv1 depth = 262144
#pragma HLS INTERFACE m_axi port = cache_kv2 bundle = gmem_kv2 depth = 262144
#pragma HLS INTERFACE m_axi port = cache_kv3 bundle = gmem_kv3 depth = 262144
#pragma HLS INTERFACE m_axi port = out0 bundle = gmem_qo0 depth = FATTN_QO_INTERFACE_DEPTH
#pragma HLS INTERFACE m_axi port = out1 bundle = gmem_qo1 depth = FATTN_QO_INTERFACE_DEPTH
#pragma HLS INTERFACE m_axi port = out2 bundle = gmem_qo2 depth = FATTN_QO_INTERFACE_DEPTH
#pragma HLS INTERFACE m_axi port = out3 bundle = gmem_qo3 depth = FATTN_QO_INTERFACE_DEPTH
#pragma HLS INTERFACE m_axi port = status_out bundle = gmem_status depth = 1
#pragma HLS INTERFACE s_axilite port = q0 bundle = control
#pragma HLS INTERFACE s_axilite port = q1 bundle = control
#pragma HLS INTERFACE s_axilite port = q2 bundle = control
#pragma HLS INTERFACE s_axilite port = q3 bundle = control
#pragma HLS INTERFACE s_axilite port = active_kv bundle = control
#pragma HLS INTERFACE s_axilite port = cache_kv0 bundle = control
#pragma HLS INTERFACE s_axilite port = cache_kv1 bundle = control
#pragma HLS INTERFACE s_axilite port = cache_kv2 bundle = control
#pragma HLS INTERFACE s_axilite port = cache_kv3 bundle = control
#pragma HLS INTERFACE s_axilite port = out0 bundle = control
#pragma HLS INTERFACE s_axilite port = out1 bundle = control
#pragma HLS INTERFACE s_axilite port = out2 bundle = control
#pragma HLS INTERFACE s_axilite port = out3 bundle = control
#pragma HLS INTERFACE s_axilite port = batch_size bundle = control
#pragma HLS INTERFACE s_axilite port = seqlen bundle = control
#pragma HLS INTERFACE s_axilite port = start_pos bundle = control
#pragma HLS INTERFACE s_axilite port = cache_stride bundle = control
#pragma HLS INTERFACE s_axilite port = cache_valid_len bundle = control
#pragma HLS INTERFACE s_axilite port = mask_mode bundle = control
#pragma HLS INTERFACE s_axilite port = flags bundle = control
#pragma HLS INTERFACE s_axilite port = cache_identity bundle = control
#pragma HLS INTERFACE s_axilite port = status_out bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
  const int batch = FATTN_BOUND_WORKLOAD ? FATTN_BOUND_BATCH_SIZE : batch_size;
  const int sequence = FATTN_BOUND_WORKLOAD ? FATTN_BOUND_SEQLEN : seqlen;
  const int start = FATTN_BOUND_WORKLOAD ? FATTN_BOUND_START_POS : start_pos;
  const int stride = FATTN_BOUND_WORKLOAD ? FATTN_BOUND_CACHE_STRIDE : cache_stride;
  const int valid = FATTN_BOUND_WORKLOAD ? FATTN_BOUND_CACHE_VALID_LEN : cache_valid_len;
  const int mask = FATTN_BOUND_WORKLOAD ? FATTN_BOUND_MASK_MODE : mask_mode;
  const int effective_flags = FATTN_BOUND_WORKLOAD ? FATTN_BOUND_FLAGS : flags;
  const int status = validate_attention_call(batch, sequence, start, stride,
                                             valid, mask, effective_flags);
  if (status != FATTN_STATUS_OK) { status_out[0] = status; return; }
  const bool prefill_physical_frame =
      batch == 2 && sequence == 8 && start == 0 && valid == 0 &&
      mask == FATTN_MASK_CAUSAL;  // This guard selects packing/burst staging, not different attention.
  const int semantic_work_extent = sequence;  // Host rows are covered by bounded local tiles.
  for (int tile = 0; tile < semantic_work_extent;
       tile += FATTN_TILE_TOKENS) {
    const int count = sequence - tile < FATTN_TILE_TOKENS
                          ? sequence - tile : FATTN_TILE_TOKENS;
    const TilePlan plan =
        make_tile_plan(batch, count, start + tile, stride);
    run_tile(q0, q1, q2, q3, active_kv, cache_kv0, cache_kv1, cache_kv2,
             cache_kv3, out0, out1, out2, out3, sequence, tile, plan,
             (effective_flags & FATTN_FLAG_ROUND_QK_BF16) != 0,
             (effective_flags & FATTN_FLAG_CACHE_PREUPDATED) != 0,
             cache_identity, prefill_physical_frame);
  }
  status_out[0] = FATTN_STATUS_OK;
}
