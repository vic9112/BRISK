//********************************************************
//* Project         : FPGenius - FPT'26 Design Competition Track B
//* Filename        : qk_datapath.hpp
//* Author          : VicChen (KUAN-HSI, CHEN)
//* Created On      : 2026/07/10
//* Function        : Implements the canonical BF16 query-key product datapath.
//********************************************************

#pragma once

#include "shared_product.hpp"
struct QPair {
  qo_pack_t words[2];
};
struct KChunk {
  bf16_t values[FATTN_VALUE_LANES];
};
struct KHalfbeat {
  KChunk halves[FATTN_HALFBEAT_CHUNKS];
};
struct Score {
  float values[FATTN_KV_REP];
};
inline float reduce_eight(const float values[FATTN_VALUE_LANES]) {  // Keep this FP32 addition tree unchanged.
#pragma HLS INLINE
  const float a0 = values[0] + values[1];
  const float a1 = values[2] + values[3];
  const float a2 = values[4] + values[5];
  const float a3 = values[6] + values[7];
  return (a0 + a1) + (a2 + a3);
}
template <int QueryOffset, int Lane>
inline void canonical_qk_product_lane(
    const qo_pack_t query[FATTN_KV_REP],
    const float key[FATTN_VALUE_LANES],
    float products[FATTN_KV_REP][FATTN_VALUE_LANES]) {
#pragma HLS INLINE
  for (int rep = 0; rep < FATTN_KV_REP; ++rep) {
#pragma HLS UNROLL
    products[rep][Lane] =
        bf16_to_float(
            qo_pack_get_static<QueryOffset + Lane>(query[rep])) *
        key[Lane];
  }
}
template <int QueryOffset>
inline void canonical_qk_products(
    const qo_pack_t query[FATTN_KV_REP],
    const float key[FATTN_VALUE_LANES],
    float products[FATTN_KV_REP][FATTN_VALUE_LANES]) {
#pragma HLS INLINE
  canonical_qk_product_lane<QueryOffset, 0>(query, key, products);
  canonical_qk_product_lane<QueryOffset, 1>(query, key, products);
  canonical_qk_product_lane<QueryOffset, 2>(query, key, products);
  canonical_qk_product_lane<QueryOffset, 3>(query, key, products);
  canonical_qk_product_lane<QueryOffset, 4>(query, key, products);
  canonical_qk_product_lane<QueryOffset, 5>(query, key, products);
  canonical_qk_product_lane<QueryOffset, 6>(query, key, products);
  canonical_qk_product_lane<QueryOffset, 7>(query, key, products);
}
// Preserve HLS source-location identifiers after separating alternatives.
#line 109
template <int Lane>
void load_q(  // Supply representative query pairs in the temporal consumer's order.
    const qo_pack_t *query, hls::stream<QPair> &rep0,
    hls::stream<QPair> &rep1, hls::stream<QPair> &rep2,
    hls::stream<QPair> &rep3, int batch_size, int total_seqlen,
    int tile_start, int tile_count, bool prefill_physical_frame) {
#pragma HLS INLINE off
  if (prefill_physical_frame) {
    qo_pack_t frame[512];  // Per batch: 8 rows * 8 local query heads * 8 packed words.
#pragma HLS BIND_STORAGE variable = frame type = ram_t2p impl = bram
    for (int batch = 0; batch < 2; ++batch) {
      for (int line = 0; line < 256; ++line) {
#pragma HLS PIPELINE II = 1
        const int base = batch * 512 + line * 2;
        frame[line * 2] = query[base];
        frame[line * 2 + 1] = query[base + 1];
      }
      for (int slot = 0; slot < FATTN_KV_SLOTS_PER_LANE; ++slot) {
        for (int pair = 0; pair < FATTN_QO_HEAD_PACKS / 2; ++pair) {
          for (int local = 0; local < 8; ++local) {
            for (int rep = 0; rep < FATTN_KV_REP; ++rep) {
#pragma HLS PIPELINE II = 1
              const int local_head = slot * FATTN_KV_REP + rep;
              const int index =
                  ((local * FATTN_Q_HEADS_PER_LANE + local_head) *
                       FATTN_QO_HEAD_PACKS +
                   pair * 2);
              QPair packet;
#pragma HLS ARRAY_PARTITION variable = packet.words complete dim = 1
              packet.words[0] = frame[index];
              packet.words[1] = frame[index + 1];
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
    return;
  }
  for (int batch = 0; batch < batch_size; ++batch) {
    for (int slot = 0; slot < FATTN_KV_SLOTS_PER_LANE; ++slot) {
      for (int pair = 0; pair < FATTN_QO_HEAD_PACKS / 2; ++pair) {
        for (int local = 0; local < tile_count; ++local) {
          const int sequence = tile_start + local;
          for (int rep = 0; rep < FATTN_KV_REP; ++rep) {
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

// Retained alternatives are not selected by the active kernel.
#include "qk_alternatives.hpp"
