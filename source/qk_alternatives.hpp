// Unselected QK alternatives retained for source compatibility.
// The kernel uses load_q and canonical_qk_products, not these functions.
#pragma once

#include "qk_datapath.hpp"

inline float fip_softmax_quotient_chunk(const float *main_products,
                                              const float *beta_products) {
#pragma HLS INLINE
  float terms[FATTN_VALUE_LANES] = {
      main_products[0], -beta_products[0],
      -main_products[1], beta_products[1],
      main_products[2], -beta_products[2],
      -main_products[3], beta_products[3]};
#pragma HLS ARRAY_PARTITION variable = terms complete dim = 1
  return reduce_eight(terms);
}
template <int Lane>
void load_q_tasks(
    const qo_pack_t *query, hls::stream<QPair> &rep0,
    hls::stream<QPair> &rep1, hls::stream<QPair> &rep2,
    hls::stream<QPair> &rep3, int batch_size, int total_seqlen,
    int tile_start, int tile_count) {
#pragma HLS INLINE off
  for (int batch = 0; batch < batch_size; ++batch) {
    for (int local = 0; local < tile_count; ++local) {
      const int sequence = tile_start + local;
      for (int slot = 0; slot < FATTN_KV_SLOTS_PER_LANE; ++slot) {
        for (int rep = 0; rep < FATTN_KV_REP; ++rep) {
          for (int pair = 0; pair < FATTN_QO_HEAD_PACKS / 2; ++pair) {
#pragma HLS PIPELINE II = 2
            const int local_head = slot * FATTN_KV_REP + rep;
            const std::size_t index =
                (((static_cast<std::size_t>(batch) * total_seqlen + sequence) *
                      FATTN_Q_HEADS_PER_LANE +
                  local_head) *
                     FATTN_QO_HEAD_PACKS +
                 pair * 2);
            QPair packet;
#pragma HLS ARRAY_PARTITION variable = packet.words complete dim = 1
            packet.words[0] = query[index];
            packet.words[1] = query[index + 1];
            switch (rep) {
              case 0: rep0.write(packet); break;
              case 1: rep1.write(packet); break;
              case 2: rep2.write(packet); break;
              default: rep3.write(packet); break;
            }
          }
        }
      }
    }
  }
}

template <int Offset, int Pair>
inline void hybrid_key_pair(
    const qo_pack_t query[FATTN_KV_REP],
    const float key[FATTN_VALUE_LANES], HybridOperands &operands) {
#pragma HLS INLINE
  constexpr int even = Pair * 2;
  constexpr int odd = even + 1;
  for (int fip_index = 0; fip_index < FATTN_HYBRID_FIP_REPS; ++fip_index) {
#pragma HLS UNROLL
    const int rep = fip_rep_from_index(fip_index);
    const float q_even = bf16_to_float(
        qo_pack_get_static<Offset + even>(query[rep]));
    const float q_odd = bf16_to_float(
        qo_pack_get_static<Offset + odd>(query[rep]));
    float left, right;
    if constexpr ((Pair & 1) == 0) {
      left = q_even + key[odd];
      right = q_odd + key[even];
    } else if constexpr (Pair < 3) {
      left = q_even - key[odd];
      right = q_odd - key[even];
    } else {
      float left_fabric, right_fabric;
#pragma HLS BIND_OP variable = left_fabric op = fsub impl = fabric
#pragma HLS BIND_OP variable = right_fabric op = fsub impl = fabric
      left_fabric = q_even - key[odd];
      right_fabric = q_odd - key[even];
      left = left_fabric;
      right = right_fabric;
    }
    const int product = fip_index * FATTN_FIP_PAIRS_PER_CHUNK + Pair;
    operands.left[product] = left;
    operands.right[product] = right;
  }
  const int beta_product = FATTN_FIP_MAIN_PRODUCTS + Pair;
  operands.left[beta_product] = key[even];
  operands.right[beta_product] = key[odd];
  for (int local_rep = 0; local_rep < FATTN_HYBRID_CLASSICAL_REPS;
       ++local_rep) {
#pragma HLS UNROLL
    const int rep = classical_rep_from_index(local_rep);
    const int product =
        FATTN_CLASSICAL_PRODUCT_BASE + local_rep * FATTN_VALUE_LANES + even;
    operands.left[product] = bf16_to_float(
        qo_pack_get_static<Offset + even>(query[rep]));
    operands.right[product] = key[even];
    operands.left[product + 1] = bf16_to_float(
        qo_pack_get_static<Offset + odd>(query[rep]));
    operands.right[product + 1] = key[odd];
  }
}
template <int Offset>
inline void hybrid_key_operands(
    const qo_pack_t query[FATTN_KV_REP],
    const float key[FATTN_VALUE_LANES], HybridOperands &operands) {
#pragma HLS INLINE
  hybrid_key_pair<Offset, 0>(query, key, operands);
  hybrid_key_pair<Offset, 1>(query, key, operands);
  hybrid_key_pair<Offset, 2>(query, key, operands);
  hybrid_key_pair<Offset, 3>(query, key, operands);
}
inline void stream_qk(
    hls::stream<QPair> &rep0, hls::stream<QPair> &rep1,
    hls::stream<QPair> &rep2, hls::stream<QPair> &rep3,
    hls::stream<KChunk> &keys, hls::stream<Score> &scores,
    int sequence_count, int context_start, int actions,
    bool round_qk_bf16) {
#pragma HLS INLINE off
  qo_pack_t q[FATTN_KV_REP][2][FATTN_QO_HEAD_PACKS / 2];
  float blocks[FATTN_KV_REP][FATTN_VALUE_CHUNKS / 2];
  float previous[FATTN_KV_REP];
#pragma HLS ARRAY_PARTITION variable = q complete dim = 1
#pragma HLS ARRAY_PARTITION variable = q complete dim = 2
#pragma HLS ARRAY_PARTITION variable = blocks complete dim = 0
#pragma HLS ARRAY_PARTITION variable = previous complete dim = 1
  int batch = 0, sequence = 0, slot = 0, position = 0, chunk = 0;
  for (int action = 0; action < actions; ++action) {
#pragma HLS PIPELINE II = 1
    if (position == 0 && chunk < FATTN_QO_HEAD_PACKS / 2) {
      const QPair p0 = rep0.read();
      const QPair p1 = rep1.read();
      const QPair p2 = rep2.read();
      const QPair p3 = rep3.read();
      q[0][0][chunk] = p0.words[0];
      q[0][1][chunk] = p0.words[1];
      q[1][0][chunk] = p1.words[0];
      q[1][1][chunk] = p1.words[1];
      q[2][0][chunk] = p2.words[0];
      q[2][1][chunk] = p2.words[1];
      q[3][0][chunk] = p3.words[0];
      q[3][1][chunk] = p3.words[1];
    }
    HybridOperands operands;
    HybridProducts products;
#pragma HLS ARRAY_PARTITION variable = operands.left complete dim = 1
#pragma HLS ARRAY_PARTITION variable = operands.right complete dim = 1
#pragma HLS ARRAY_PARTITION variable = products.values complete dim = 1
    const int q_pack =
        (chunk * FATTN_VALUE_LANES) / FATTN_QO_PACK_ELEMS;
    const int bank = q_pack & 1;
    const int index = q_pack >> 1;
    qo_pack_t query_words[FATTN_KV_REP] = {
        q[0][bank][index], q[1][bank][index],
        q[2][bank][index], q[3][bank][index]};
#pragma HLS ARRAY_PARTITION variable = query_words complete dim = 1
    const KChunk key_chunk = keys.read();
    float key[FATTN_VALUE_LANES];
#pragma HLS ARRAY_PARTITION variable = key complete dim = 1
    for (int lane = 0; lane < FATTN_VALUE_LANES; ++lane) {
#pragma HLS UNROLL
      key[lane] = bf16_to_float(key_chunk.values[lane]);
    }
    if ((chunk & 1) == 0)
      hybrid_key_operands<0>(query_words, key, operands);
    else
      hybrid_key_operands<8>(query_words, key, operands);
    hybrid_product_bank(operands, products);
    const float *beta_products =
        &products.values[FATTN_FIP_MAIN_PRODUCTS];
    for (int rep = 0; rep < FATTN_KV_REP; ++rep) {
#pragma HLS UNROLL
      float sum;
      if (rep_uses_fip(rep)) {
        const int fip_index = fip_rep_index(rep);
        sum = fip_softmax_quotient_chunk(
            &products.values[fip_index * FATTN_FIP_PAIRS_PER_CHUNK],
            beta_products);
      } else {
        const int local_rep = classical_rep_index(rep);
        sum = reduce_eight(
            &products.values[FATTN_CLASSICAL_PRODUCT_BASE +
                             local_rep * FATTN_VALUE_LANES]);
      }
      if ((chunk & 1) == 0)
        previous[rep] = sum;
      else
        blocks[rep][chunk >> 1] = previous[rep] + sum;
    }
    if (++chunk == FATTN_VALUE_CHUNKS) {
      Score result;
#pragma HLS ARRAY_PARTITION variable = result.values complete dim = 1
      for (int rep = 0; rep < FATTN_KV_REP; ++rep) {
#pragma HLS UNROLL
        const float score = reduce_eight(blocks[rep]);
        result.values[rep] = maybe_round_bf16(
            score * FATTN_INV_SQRT_HEAD_DIM, round_qk_bf16);
      }
      scores.write(result);
      chunk = 0;
      if (++position == context_start + sequence + 1) {
        position = 0;
        if (++slot == FATTN_KV_SLOTS_PER_LANE) {
          slot = 0;
          if (++sequence == sequence_count) {
            sequence = 0;
            ++batch;
          }
        }
      }
    }
  }
}
