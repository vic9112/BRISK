// Unselected streaming alternatives retained in the original unnamed namespace.
// The active kernel uses load_kv and sv_lane instead of these functions.
#pragma once

#include "attention_tile.hpp"

namespace {
using ActiveTile =
    kv_pair_pack_t[FATTN_KV_SLOTS_PER_LANE][FATTN_MAX_BATCH]
                        [FATTN_TILE_TOKENS][FATTN_KV_HEAD_PACKS];

inline std::size_t cache_index(int batch, int position, int slot, int line,
                               int stride) {
#pragma HLS INLINE
  return (((static_cast<std::size_t>(batch) * stride + position) *
                FATTN_KV_SLOTS_PER_LANE +
            slot) *
               FATTN_CACHE_LINES_PER_SLOT +
           line);
}

template <int Lane>
inline void unpack_kv_lane(kv_pair_pack_t word, KChunk &key,
                           VChunk &value) {
#pragma HLS INLINE
  key.values[Lane] = kv_pair_pack_get_k_static<Lane>(word);
  value.values[Lane] = kv_pair_pack_get_v_static<Lane>(word);
}
inline void unpack_kv(kv_pair_pack_t word, KChunk &key,
                      VChunk &value) {
#pragma HLS INLINE
  unpack_kv_lane<0>(word, key, value); unpack_kv_lane<1>(word, key, value);
  unpack_kv_lane<2>(word, key, value); unpack_kv_lane<3>(word, key, value);
  unpack_kv_lane<4>(word, key, value); unpack_kv_lane<5>(word, key, value);
  unpack_kv_lane<6>(word, key, value); unpack_kv_lane<7>(word, key, value);
}
void stream_cache_tile(
    const cache_line_t *cache, hls::stream<bool> &ready,
    hls::stream<cache_line_t> &forwarded,
    hls::stream<KChunk> &keys, hls::stream<VChunk> &values,
    int sequences, int start, int cache_stride, int actions) {
#pragma HLS INLINE off
  ActiveTile active;
#pragma HLS ARRAY_PARTITION variable = active complete dim = 1
#pragma HLS BIND_STORAGE variable = active type = ram_2p impl = bram
  cache_line_t active_line = 0;
  int batch = 0, sequence = 0, slot = 0, position = 0, chunk = 0;
  for (int action = 0; action < actions; ++action) {
#pragma HLS PIPELINE II = 1
    if (position == 0 && chunk == 0) (void)ready.read();
    const int pack = chunk;
    kv_pair_pack_t word;
    if (position >= start) {
      const int active_sequence = position - start;
      if (active_sequence == sequence) {
        if ((pack & 1) == 0) active_line = forwarded.read();
        word = cache_line_get(active_line, pack & 1);
        active[slot][batch][active_sequence][pack] = word;
      } else {
        word = active[slot][batch][active_sequence][pack];
      }
    } else {
      const cache_line_t line =
          cache[cache_index(batch, position, slot, pack >> 1, cache_stride)];
      word = cache_line_get(line, pack & 1);
    }
    KChunk key;
    VChunk value;
#pragma HLS ARRAY_PARTITION variable = key.values complete dim = 1
#pragma HLS ARRAY_PARTITION variable = value.values complete dim = 1
    unpack_kv(word, key, value);
    keys.write(key);
    values.write(value);
    if (++chunk == FATTN_VALUE_CHUNKS) {
      chunk = 0;
      if (++position == start + sequence + 1) {
        position = 0;
        if (++slot == FATTN_KV_SLOTS_PER_LANE) {
          slot = 0;
          if (++sequence == sequences) {
            sequence = 0;
            ++batch;
          }
        }
      }
    }
  }
}
void accumulate_values(
    hls::stream<Scale> &scales, hls::stream<VChunk> &values,
    hls::stream<AccChunk> &results, int sequences, int start,
    int actions) {
#pragma HLS INLINE off
  float state[FATTN_KV_REP][FATTN_HEAD_DIM];
#pragma HLS ARRAY_PARTITION variable = state complete dim = 1
#pragma HLS ARRAY_PARTITION variable = state cyclic factor = FATTN_VALUE_LANES dim = 2
  Scale scale;
  int batch = 0, sequence = 0, slot = 0, position = 0, chunk = 0;
  for (int action = 0; action < actions; ++action) {
#pragma HLS DEPENDENCE variable = state inter false
#pragma HLS PIPELINE II = 1
    if (chunk == 0) scale = scales.read();
    const VChunk value = values.read();
    AccChunk result;
#pragma HLS ARRAY_PARTITION variable = value.values complete dim = 1
#pragma HLS ARRAY_PARTITION variable = result.values complete dim = 0
    if (position == 0) {
      for (int lane = 0; lane < FATTN_VALUE_LANES; ++lane) {
#pragma HLS UNROLL
        const int dimension = chunk * FATTN_VALUE_LANES + lane;
        const float v = bf16_to_float(value.values[lane]);
        for (int rep = 0; rep < FATTN_KV_REP; ++rep) {
#pragma HLS UNROLL
          state[rep][dimension] = v;
          result.values[rep][lane] = v;
        }
      }
    } else {
      for (int lane = 0; lane < FATTN_VALUE_LANES; ++lane) {
#pragma HLS UNROLL
        const int dimension = chunk * FATTN_VALUE_LANES + lane;
        const float v = bf16_to_float(value.values[lane]);
        for (int rep = 0; rep < FATTN_KV_REP; ++rep) {
#pragma HLS UNROLL
          const float previous = state[rep][dimension];
          const bool grew = scale_max_grew(scale, rep);
          const float multiply = grew ? previous : v;
          const float factor = scale_nontrivial(scale, rep);
          const float add = grew ? v : previous;
          const float next = multiply * factor + add;
          state[rep][dimension] = next;
          result.values[rep][lane] = next;
        }
      }
    }
    if (position + 1 == start + sequence + 1) results.write(result);
    if (++chunk == FATTN_VALUE_CHUNKS) {
      chunk = 0;
      if (++position == start + sequence + 1) {
        position = 0;
        if (++slot == FATTN_KV_SLOTS_PER_LANE) {
          slot = 0;
          if (++sequence == sequences) {
            sequence = 0;
            ++batch;
          }
        }
      }
    }
  }
}
}  // namespace
