// Unselected KV residency/update alternatives retained for source compatibility.
// The kernel uses load_kv and sv_lane instead of these functions.
#pragma once

#include "kv_stream.hpp"

template <int Lane>
inline void load_resident_kv(
    const cache_line_t *cache, hls::stream<bool> &ready,
    hls::stream<cache_line_t> &forwarded,
    hls::stream<KChunk> &keys, hls::stream<VChunk> &values,
    int batches, int sequences, int start, int cache_stride) {
#pragma HLS INLINE off
  static cache_line_t hot_cache[FATTN_PERSISTENT_HOT_KV_TOKENS]
                                      [FATTN_HOT_LINES];
  static int hot_expected_start = 0;
  static int hot_cache_stride = 0;
  static int hot_batches = 0;
  static bool hot_state_valid = false;
#pragma HLS BIND_STORAGE variable = hot_cache type = ram_2p impl = uram
  const bool continuous =
      hot_state_valid && start != 0 && start == hot_expected_start &&
      cache_stride == hot_cache_stride && batches == hot_batches;
  const int hot_tokens_per_batch =
      batches == 1 ? FATTN_PERSISTENT_HOT_KV_TOKENS
                   : FATTN_PERSISTENT_HOT_KV_TOKENS / FATTN_MAX_BATCH;
  const int hot_resident_start =
      start > hot_tokens_per_batch ? start - hot_tokens_per_batch : 0;
  KeyStationaryActive active;
#pragma HLS ARRAY_PARTITION variable = active complete dim = 1
#pragma HLS BIND_STORAGE variable = active type = ram_2p impl = bram
  for (int batch = 0; batch < batches; ++batch)
    for (int row = 0; row < sequences; ++row)
      for (int slot = 0; slot < FATTN_KV_SLOTS_PER_LANE; ++slot) {
        (void)ready.read();
        for (int line = 0; line < FATTN_CACHE_LINES_PER_SLOT; ++line) {
#pragma HLS PIPELINE II = 1
          const cache_line_t active_line = forwarded.read();
          active[slot][batch][row][line * 2] =
              cache_line_get(active_line, 0);
          active[slot][batch][row][line * 2 + 1] =
              cache_line_get(active_line, 1);
        }
      }
  KChunk key_buffer[FATTN_VALUE_CHUNKS];
  VChunk value_buffer[FATTN_VALUE_CHUNKS];
#pragma HLS BIND_STORAGE variable = key_buffer type = ram_2p impl = bram
#pragma HLS BIND_STORAGE variable = value_buffer type = ram_2p impl = bram
  for (int batch = 0; batch < batches; ++batch) {
    for (int position = 0; position < start + sequences; ++position) {
      const int first_row = position < start ? 0 : position - start;
      const int hot_index =
          batches == 1
              ? position & (FATTN_PERSISTENT_HOT_KV_TOKENS - 1)
              : batch * (FATTN_PERSISTENT_HOT_KV_TOKENS / FATTN_MAX_BATCH) +
                    (position &
                     (FATTN_PERSISTENT_HOT_KV_TOKENS / FATTN_MAX_BATCH - 1));
      const bool hot_hit = continuous && position < start &&
                           position >= hot_resident_start;
      for (int slot = 0; slot < FATTN_KV_SLOTS_PER_LANE; ++slot) {
        for (int line = 0; line < FATTN_CACHE_LINES_PER_SLOT; ++line) {
#pragma HLS PIPELINE II = 1
          const int hot_line =
              slot * FATTN_CACHE_LINES_PER_SLOT + line;
          const cache_line_t cache_line =
              position < start
                  ? (hot_hit
                         ? hot_cache[hot_index][hot_line]
                         : cache[kv_cache_index(
                               batch, position, slot, line, cache_stride)])
                  : make_cache_line(
                        active[slot][batch][position - start][line * 2],
                        active[slot][batch][position - start][line * 2 + 1]);
          for (int half = 0; half < FATTN_CACHE_LINE_PACKS; ++half) {
#pragma HLS UNROLL
            const int chunk = line * FATTN_CACHE_LINE_PACKS + half;
            const kv_pair_pack_t word =
                position < start
                    ? cache_line_get(cache_line, half)
                    : active[slot][batch][position - start][chunk];
            KChunk key;
            VChunk value;
#pragma HLS ARRAY_PARTITION variable = key.values complete dim = 1
#pragma HLS ARRAY_PARTITION variable = value.values complete dim = 1
            kv_unpack(word, key, value);
            key_buffer[chunk] = key;
            value_buffer[chunk] = value;
          }
          if (!hot_hit && (!continuous || position >= start))
            hot_cache[hot_index][hot_line] = cache_line;
        }
        for (int row = first_row; row < sequences; ++row) {
          for (int chunk = 0; chunk < FATTN_VALUE_CHUNKS; ++chunk) {
#pragma HLS PIPELINE II = 1
            keys.write(key_buffer[chunk]);
            values.write(value_buffer[chunk]);
          }
        }
      }
    }
  }
  hot_expected_start = start + sequences;
  hot_cache_stride = cache_stride;
  hot_batches = batches;
  hot_state_valid = true;
}

inline void sv_update(
    hls::stream<Scale> &scales, hls::stream<VChunk> &values,
    hls::stream<AccChunk> &results, int batches, int sequences,
    int start) {
#pragma HLS INLINE off
  float state[FATTN_TILE_TOKENS][FATTN_KV_SLOTS_PER_LANE]
             [FATTN_KV_REP][FATTN_HEAD_DIM];
#pragma HLS ARRAY_PARTITION variable = state complete dim = 3
#pragma HLS ARRAY_PARTITION variable = state cyclic factor = FATTN_VALUE_LANES dim = 4
#pragma HLS BIND_STORAGE variable = state type = ram_t2p impl = bram
  const int visible_rows =
      sequences * start + sequences * (sequences + 1) / 2;
  const int total_actions =
      batches * visible_rows * FATTN_KV_SLOTS_PER_LANE * FATTN_VALUE_CHUNKS;
  int position = 0, slot = 0, row = 0, chunk = 0;
  Scale scale;
  for (int action = 0; action < total_actions; ++action) {
#pragma HLS DEPENDENCE variable = state inter false
#pragma HLS PIPELINE II = 1
    if (chunk == 0) scale = scales.read();
    const VChunk value = values.read();
    AccChunk result;
#pragma HLS ARRAY_PARTITION variable = value.values complete dim = 1
#pragma HLS ARRAY_PARTITION variable = result.values complete dim = 0
    for (int lane = 0; lane < FATTN_VALUE_LANES; ++lane) {
#pragma HLS UNROLL
      const int dimension = chunk * FATTN_VALUE_LANES + lane;
      const float current = bf16_to_float(value.values[lane]);
      for (int rep = 0; rep < FATTN_KV_REP; ++rep) {
#pragma HLS UNROLL
        float next;
        if (position == 0) {
          next = current;
        } else {
          const float previous = state[row][slot][rep][dimension];
          const bool grew = scale_max_grew(scale, rep);
          const float multiply = grew ? previous : current;
          const float factor = scale_nontrivial(scale, rep);
          const float add = grew ? current : previous;
          next = multiply * factor + add;
        }
        state[row][slot][rep][dimension] = next;
        result.values[rep][lane] = next;
      }
    }
    if (position == start + row) results.write(result);
    if (++chunk == FATTN_VALUE_CHUNKS) {
      chunk = 0;
      if (++row == sequences) {
        if (++slot == FATTN_KV_SLOTS_PER_LANE) {
          slot = 0;
          if (++position == start + sequences) position = 0;
        }
        row = position < start ? 0 : position - start;
      }
    }
  }
}
