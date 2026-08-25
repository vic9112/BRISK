//********************************************************
//* Project         : FPGenius - FPT'26 Design Competition Track B
//* Filename        : kv_stream.hpp
//* Author          : VicChen (KUAN-HSI, CHEN)
//* Created On      : 2026/07/10
//* Function        : Streams resident and fallback KV data through fused score, softmax, and value updates.
//********************************************************

#pragma once

#include "attention_tile.hpp"
using KeyStationaryActive =
    kv_pair_pack_t[FATTN_KV_SLOTS_PER_LANE][FATTN_MAX_BATCH]
                        [FATTN_TILE_TOKENS][FATTN_KV_HEAD_PACKS];
constexpr int FATTN_KV_STATIONARY_SLOTS = 1280;
constexpr int FATTN_SLOTS_PER_LAYER = 40;
constexpr int FATTN_KV_STATIONARY_B2_WINDOW = 20;
constexpr int FATTN_KV_STATIONARY_B1_WINDOW = 40;
constexpr int FATTN_PERSISTENT_HOT_KV_TOKENS =
    FATTN_KV_STATIONARY_SLOTS;
constexpr int FATTN_HOT_LINES =
    FATTN_KV_SLOTS_PER_LANE * FATTN_CACHE_LINES_PER_SLOT;
static_assert(
    FATTN_PERSISTENT_HOT_KV_TOKENS ==
        32 * FATTN_SLOTS_PER_LAYER,
    "KV-stationary storage requires forty slots per Llama layer");
constexpr int FATTN_LIPR_LAYER_BITS = 5;
constexpr int FATTN_LIPR_LAYER_MASK = (1 << FATTN_LIPR_LAYER_BITS) - 1;
constexpr int FATTN_LIPR_TAG_BITS = 49;
struct KvTag {
  ap_uint<32> cache_identity;
  ap_uint<13> position;
  ap_uint<1> batch;
  ap_uint<1> valid;
  ap_uint<1> dirty;
  ap_uint<1> two_batches;
};
struct KvStationaryState {
  ap_uint<32> cache_identity;
  ap_uint<14> expected_start;
  ap_uint<1> two_batches;
  ap_uint<1> valid;
};
inline int kv_position_slot(int position, int window) {
#pragma HLS INLINE
  return position % window;
}
inline int kv_index_from_slot(
    int cache_identity, int batch, int position_slot, int batches) {
#pragma HLS INLINE
  const int layer_id = cache_identity & FATTN_LIPR_LAYER_MASK;
  const int layer_base =
      layer_id * FATTN_SLOTS_PER_LAYER;
  return batches == 1
             ? layer_base + position_slot
             : layer_base +
                   batch * FATTN_KV_STATIONARY_B2_WINDOW +
                   position_slot;
}
inline void kv_advance_position_slot(
    int &position_slot, int batches) {
#pragma HLS INLINE
  const int window =
      batches == 1 ? FATTN_KV_STATIONARY_B1_WINDOW
                   : FATTN_KV_STATIONARY_B2_WINDOW;
  if (++position_slot == window) position_slot = 0;
}
inline int kv_index(
    int cache_identity, int batch, int position, int batches) {
#pragma HLS INLINE
  const int layer_id = cache_identity & FATTN_LIPR_LAYER_MASK;
  const int layer_base =
      layer_id * FATTN_SLOTS_PER_LAYER;
  if (batches == 1) {
    return layer_base + kv_position_slot(
                            position,
                            FATTN_KV_STATIONARY_B1_WINDOW);
  }
  return layer_base + batch * FATTN_KV_STATIONARY_B2_WINDOW +
         kv_position_slot(
             position, FATTN_KV_STATIONARY_B2_WINDOW);
}
inline KvTag kv_tag(
    int cache_identity, int batch, int position, bool dirty = false
    , int batches = 1
    ) {
#pragma HLS INLINE
  KvTag tag;
  tag.cache_identity = cache_identity;
  tag.position = position;
  tag.batch = batch;
  tag.valid = 1;
  tag.dirty = dirty;
  tag.two_batches = batches == 2;
  return tag;
}
inline bool kv_geometry_matches(
    const KvTag &tag, int batches) {
#pragma HLS INLINE
  return tag.two_batches == (batches == 2);
}
inline std::size_t kv_cache_index(
    int batch, int position, int slot, int line, int stride) {
#pragma HLS INLINE
  return (((static_cast<std::size_t>(batch) * stride + position) *
               FATTN_KV_SLOTS_PER_LANE +
           slot) *
              FATTN_CACHE_LINES_PER_SLOT +
          line);
}
inline std::size_t dual_port_cache_index(
    int batch, int position, int slot, int line, int stride) {
#pragma HLS INLINE
  return (((static_cast<std::size_t>(batch) * stride + position) *
               FATTN_KV_SLOTS_PER_LANE +
           slot) *
              FATTN_PLANE_LINES +
          line);
}
template <int Lane>
inline void kv_unpack_lane(
    kv_pair_pack_t word, KChunk &key, VChunk &value) {
#pragma HLS INLINE
  key.values[Lane] = kv_pair_pack_get_k_static<Lane>(word);
  value.values[Lane] = kv_pair_pack_get_v_static<Lane>(word);
}
inline void kv_unpack(
    kv_pair_pack_t word, KChunk &key, VChunk &value) {
#pragma HLS INLINE
  kv_unpack_lane<0>(word, key, value);
  kv_unpack_lane<1>(word, key, value);
  kv_unpack_lane<2>(word, key, value);
  kv_unpack_lane<3>(word, key, value);
  kv_unpack_lane<4>(word, key, value);
  kv_unpack_lane<5>(word, key, value);
  kv_unpack_lane<6>(word, key, value);
  kv_unpack_lane<7>(word, key, value);
}
template <int Lane>
inline bf16_t dim_pack_get_static(kv_pack_t pack) {
#pragma HLS INLINE
  return static_cast<bf16_t>(
      pack.range(Lane * 16 + 15, Lane * 16));
}
inline KChunk dim_key_chunk(kv_pack_t pack) {
#pragma HLS INLINE
  KChunk key;
#pragma HLS ARRAY_PARTITION variable = key.values complete dim = 1
  key.values[0] = dim_pack_get_static<0>(pack);
  key.values[1] = dim_pack_get_static<1>(pack);
  key.values[2] = dim_pack_get_static<2>(pack);
  key.values[3] = dim_pack_get_static<3>(pack);
  key.values[4] = dim_pack_get_static<4>(pack);
  key.values[5] = dim_pack_get_static<5>(pack);
  key.values[6] = dim_pack_get_static<6>(pack);
  key.values[7] = dim_pack_get_static<7>(pack);
  return key;
}
inline VChunk dim_value_chunk(kv_pack_t pack) {
#pragma HLS INLINE
  VChunk value;
#pragma HLS ARRAY_PARTITION variable = value.values complete dim = 1
  value.values[0] = dim_pack_get_static<0>(pack);
  value.values[1] = dim_pack_get_static<1>(pack);
  value.values[2] = dim_pack_get_static<2>(pack);
  value.values[3] = dim_pack_get_static<3>(pack);
  value.values[4] = dim_pack_get_static<4>(pack);
  value.values[5] = dim_pack_get_static<5>(pack);
  value.values[6] = dim_pack_get_static<6>(pack);
  value.values[7] = dim_pack_get_static<7>(pack);
  return value;
}
template <int Lane>
inline void load_kv(
    cache_line_t *cache, hls::stream<bool> &ready,
    hls::stream<cache_line_t> &forwarded,
    hls::stream<DimensionKPacket> &keys,
    hls::stream<DimensionVPacket> &values,
    int batches, int sequences, int start, int cache_stride,
    int cache_identity, bool cache_preupdated) {
#pragma HLS INLINE off
  (void)ready;
  static cache_line_t hot_cache[FATTN_PERSISTENT_HOT_KV_TOKENS]
                                      [FATTN_HOT_LINES];
#define FATTN_HOT_CACHE_LINE(index,line) hot_cache[(index)][(line)]
  static KvTag hot_tags[FATTN_PERSISTENT_HOT_KV_TOKENS] = {};
  static KvStationaryState wavefront_states[32] = {};
#pragma HLS ARRAY_PARTITION variable = hot_cache block factor = 5 dim = 1
#pragma HLS BIND_STORAGE variable = hot_cache type = ram_2p impl = uram
#pragma HLS DEPENDENCE variable = hot_cache inter false
#pragma HLS BIND_STORAGE variable = hot_tags type = ram_2p impl = bram
#pragma HLS AGGREGATE variable = hot_tags compact = bit
#pragma HLS DEPENDENCE variable = hot_tags inter false
#pragma HLS BIND_STORAGE variable = wavefront_states type = ram_2p impl = bram
#pragma HLS AGGREGATE variable = wavefront_states compact = bit
  const int visible_rows =
      sequences * start + sequences * (sequences + 1) / 2;
  const int total_actions =
      batches * visible_rows * FATTN_KV_SLOTS_PER_LANE *
      FATTN_VALUE_CHUNKS;
  const int wavefront_layer = cache_identity & FATTN_LIPR_LAYER_MASK;
  const int wavefront_window =
      batches == 1 ? FATTN_KV_STATIONARY_B1_WINDOW
                   : FATTN_KV_STATIONARY_B2_WINDOW;
  const KvStationaryState wavefront_state =
      wavefront_states[wavefront_layer];
  const bool wavefront_continuation =
      start == 0 ||
      (wavefront_state.valid && start > 0 &&
       wavefront_state.cache_identity == cache_identity &&
       wavefront_state.expected_start == start &&
       wavefront_state.two_batches == (batches == 2));
  const bool resident_fast_path =
      wavefront_continuation &&
      start + sequences <= wavefront_window;
  if (resident_fast_path) {
    cache_line_t resident_line_buffer = 0;
    int resident_batch = 0;
    int resident_position = 0;
    int resident_slot = 0;
    int resident_chunk = 0;
    int resident_row = 0;
    int resident_position_slot = 0;
resident_action_loop:
    for (int resident_action = 0;
         resident_action < total_actions; ++resident_action) {
#pragma HLS PIPELINE II = 1
      const int resident_first_row =
          resident_position < start ? 0 : resident_position - start;
      const int resident_hot_index =
          kv_index_from_slot(
              cache_identity, resident_batch,
              resident_position_slot, batches);
      const int resident_storage_line =
          resident_chunk / FATTN_CACHE_LINE_PACKS;
      if ((resident_chunk % FATTN_CACHE_LINE_PACKS) == 0 &&
          resident_row == resident_first_row) {
        const int resident_hot_line =
            resident_slot * FATTN_CACHE_LINES_PER_SLOT +
            resident_storage_line;
        if (resident_position < start) {
          resident_line_buffer = hot_cache[(resident_hot_index)][(resident_hot_line)];
        } else {
          resident_line_buffer = forwarded.read();
          hot_cache[(resident_hot_index)][(resident_hot_line)] =
              resident_line_buffer;
          if (resident_slot + 1 == FATTN_KV_SLOTS_PER_LANE &&
              resident_storage_line + 1 ==
                  FATTN_CACHE_LINES_PER_SLOT) {
            hot_tags[resident_hot_index] = kv_tag(
                cache_identity, resident_batch,
                resident_position, !cache_preupdated, batches);
          }
        }
      }
      KChunk resident_key;
      VChunk resident_value;
#pragma HLS ARRAY_PARTITION variable = resident_key.values complete dim = 1
#pragma HLS ARRAY_PARTITION variable = resident_value.values complete dim = 1
      kv_unpack(
          cache_line_get(
              resident_line_buffer,
              resident_chunk % FATTN_CACHE_LINE_PACKS),
          resident_key, resident_value);
      if (resident_row == resident_first_row)
        keys.write(resident_key);
      if (resident_row == resident_first_row)
        values.write(resident_value);
      if (++resident_row == sequences) {
        resident_row = resident_first_row;
        if (++resident_chunk == FATTN_VALUE_CHUNKS) {
          resident_chunk = 0;
          if (++resident_slot == FATTN_KV_SLOTS_PER_LANE) {
            resident_slot = 0;
            ++resident_position;
            kv_advance_position_slot(
                resident_position_slot, batches);
            if (resident_position == start + sequences) {
              resident_position = 0;
              resident_position_slot = 0;
              ++resident_batch;
            }
          }
        }
        resident_row =
            resident_position < start
                ? 0
                : resident_position - start;
      }
    }
  } else {
  cache_line_t line_buffer = 0;
  bool hot_hit = false;
  int batch = 0;
  int position = 0;
  int slot = 0;
  int chunk = 0;
  int row = 0;
  int kv_position_slot = 0;
cache_action_loop:
  for (int action = 0; action < total_actions; ++action) {
#pragma HLS PIPELINE II = 1
    const int first_row = position < start ? 0 : position - start;
    const int hot_index =
        kv_index_from_slot(
            cache_identity, batch, kv_position_slot, batches);
    if (slot == 0 && chunk == 0 && row == first_row) {
      hot_hit =
          position < start && hot_tags[hot_index].valid &&
          hot_tags[hot_index].cache_identity == cache_identity &&
          hot_tags[hot_index].batch == batch &&
          hot_tags[hot_index].position == position
          && kv_geometry_matches(hot_tags[hot_index], batches)
          ;
    }
    const int storage_line = chunk / FATTN_CACHE_LINE_PACKS;
    if ((chunk % FATTN_CACHE_LINE_PACKS) == 0 && row == first_row) {
      const int hot_line =
          slot * FATTN_CACHE_LINES_PER_SLOT + storage_line;
      bool update_hot = false;
      const KvTag kv_victim_tag = hot_tags[hot_index];
      const bool kv_writeback =
          !hot_hit && kv_victim_tag.valid &&
          kv_victim_tag.dirty &&
          kv_victim_tag.cache_identity == cache_identity &&
          kv_geometry_matches(
              kv_victim_tag, batches);
      if (kv_writeback) {
        cache[kv_cache_index(
            kv_victim_tag.batch,
            kv_victim_tag.position, slot, storage_line,
            cache_stride)] =
            hot_cache[(hot_index)][(hot_line)];
      }
      if (position < start) {
        line_buffer =
            hot_hit
                ? hot_cache[(hot_index)][(hot_line)]
                : cache[kv_cache_index(
                      batch, position, slot, storage_line, cache_stride)];
        update_hot = !hot_hit;
      } else {
        line_buffer = forwarded.read();
        update_hot = true;
      }
      if (update_hot)
        hot_cache[(hot_index)][(hot_line)] = line_buffer;
      if (update_hot && slot + 1 == FATTN_KV_SLOTS_PER_LANE &&
          storage_line + 1 == FATTN_CACHE_LINES_PER_SLOT) {
        hot_tags[hot_index] =
            kv_tag(
                cache_identity, batch, position,
                position >= start && !cache_preupdated, batches
            );
      }
    }
    KChunk key;
    VChunk value;
#pragma HLS ARRAY_PARTITION variable = key.values complete dim = 1
#pragma HLS ARRAY_PARTITION variable = value.values complete dim = 1
    kv_unpack(
        cache_line_get(
            line_buffer, chunk % FATTN_CACHE_LINE_PACKS),
        key, value);
    if (row == first_row) keys.write(key);
    if (row == first_row) values.write(value);
    if (++row == sequences) {
      row = first_row;
      if (++chunk == FATTN_VALUE_CHUNKS) {
        chunk = 0;
        if (++slot == FATTN_KV_SLOTS_PER_LANE) {
          slot = 0;
          ++position;
          kv_advance_position_slot(
              kv_position_slot, batches);
          if (position == start + sequences) {
            position = 0;
            kv_position_slot = 0;
            ++batch;
          }
        }
      }
      row = position < start ? 0 : position - start;
    }
  }
  }
  KvStationaryState next_wavefront_state;
  next_wavefront_state.cache_identity = cache_identity;
  next_wavefront_state.expected_start = start + sequences;
  next_wavefront_state.two_batches = batches == 2;
  next_wavefront_state.valid = 1;
  wavefront_states[wavefront_layer] = next_wavefront_state;
  (void)cache_preupdated;
#undef FATTN_HOT_CACHE_LINE
}
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
inline void qk_lane(
    hls::stream<QPair> &rep0, hls::stream<QPair> &rep1,
    hls::stream<QPair> &rep2, hls::stream<QPair> &rep3,
    hls::stream<KChunk> &keys, hls::stream<Score> &scores,
    int batches, int sequences, int context_start, int actions,
    bool round_qk_bf16) {
#pragma HLS INLINE off
  qo_pack_t query[FATTN_MAX_BATCH][FATTN_TILE_TOKENS]
                       [FATTN_KV_SLOTS_PER_LANE][FATTN_KV_REP]
                       [FATTN_QO_HEAD_PACKS];
#pragma HLS ARRAY_PARTITION variable = query complete dim = 4
#pragma HLS BIND_STORAGE variable = query type = ram_2p impl = bram
  float blocks[FATTN_TILE_TOKENS][FATTN_KV_SLOTS_PER_LANE]
              [FATTN_KV_REP][FATTN_VALUE_CHUNKS / 2];
  float previous[FATTN_TILE_TOKENS][FATTN_KV_SLOTS_PER_LANE]
                [FATTN_KV_REP];
  KChunk held_key;
  float held_beta[FATTN_FIP_BETA_PRODUCTS];
#pragma HLS ARRAY_PARTITION variable = blocks complete dim = 3
#pragma HLS ARRAY_PARTITION variable = previous complete dim = 3
#pragma HLS ARRAY_PARTITION variable = held_key.values complete dim = 1
#pragma HLS ARRAY_PARTITION variable = held_beta complete dim = 1
#pragma HLS BIND_STORAGE variable = blocks type = ram_t2p impl = bram
#pragma HLS BIND_STORAGE variable = previous type = ram_t2p impl = bram
  int batch = 0, position = 0, slot = 0, chunk = 0, row = 0;
qk_action_loop:
  for (int action = 0; action < actions; ++action) {
#pragma HLS PIPELINE II = 1 style = flp
          const int first_row =
              position < context_start ? 0 : position - context_start;
          qo_pack_t query_words[FATTN_KV_REP];
          QPair packets[FATTN_KV_REP];
          float key_values[FATTN_VALUE_LANES];
          float canonical_products[FATTN_KV_REP][FATTN_VALUE_LANES];
#pragma HLS ARRAY_PARTITION variable = query_words complete dim = 1
#pragma HLS ARRAY_PARTITION variable = packets complete dim = 0
#pragma HLS ARRAY_PARTITION variable = key_values complete dim = 1
#pragma HLS ARRAY_PARTITION variable = canonical_products complete dim = 0
          const bool load_query = position == 0 && (chunk & 3) == 0;
          if (row == first_row) {
            held_key = keys.read();
          }
          const KChunk key = held_key;
          if (load_query) {
            packets[0] = rep0.read();
            packets[1] = rep1.read();
            packets[2] = rep2.read();
            packets[3] = rep3.read();
          }
          for (int rep = 0; rep < FATTN_KV_REP; ++rep) {
#pragma HLS UNROLL
            if (load_query) {
              query[batch][row][slot][rep][chunk >> 1] =
                  packets[rep].words[0];
              query[batch][row][slot][rep][(chunk >> 1) + 1] =
                  packets[rep].words[1];
              query_words[rep] = packets[rep].words[0];
            } else {
              query_words[rep] =
                  query[batch][row][slot][rep][chunk >> 1];
            }
          }
          for (int lane = 0; lane < FATTN_VALUE_LANES; ++lane) {
#pragma HLS UNROLL
            key_values[lane] = bf16_to_float(key.values[lane]);
          }
          if ((chunk & 1) == 0)
            canonical_qk_products<0>(
                query_words, key_values, canonical_products);
          else
            canonical_qk_products<8>(
                query_words, key_values, canonical_products);
          Score result;
#pragma HLS ARRAY_PARTITION variable = result.values complete dim = 1
          for (int rep = 0; rep < FATTN_KV_REP; ++rep) {
#pragma HLS UNROLL
            float chunk_sum;
            chunk_sum = reduce_eight(canonical_products[rep]);
            if ((chunk & 1) == 0) {
              previous[row][slot][rep] = chunk_sum;
            } else {
              const float block_sum =
                  previous[row][slot][rep] + chunk_sum;
              blocks[row][slot][rep][chunk >> 1] = block_sum;
              if (chunk + 1 == FATTN_VALUE_CHUNKS) {
                float complete_blocks[FATTN_VALUE_CHUNKS / 2];
#pragma HLS ARRAY_PARTITION variable = complete_blocks complete dim = 1
                for (int block = 0;
                     block < FATTN_VALUE_CHUNKS / 2; ++block) {
#pragma HLS UNROLL
                  complete_blocks[block] =
                      block + 1 == FATTN_VALUE_CHUNKS / 2
                          ? block_sum
                          : blocks[row][slot][rep][block];
                }
                const float score = reduce_eight(complete_blocks);
                result.values[rep] = maybe_round_bf16(
                    score * FATTN_INV_SQRT_HEAD_DIM, round_qk_bf16);
              }
            }
          }
          if (chunk + 1 == FATTN_VALUE_CHUNKS) scores.write(result);
          if (++row == sequences) {
            row = first_row;
            if (++chunk == FATTN_VALUE_CHUNKS) {
              chunk = 0;
              if (++slot == FATTN_KV_SLOTS_PER_LANE) {
                slot = 0;
                if (++position == context_start + sequences) {
                  position = 0;
                  ++batch;
                }
              }
              row =
                  position < context_start ? 0 : position - context_start;
            }
          }
  }
  (void)batches;
}
inline void online_sm_step(
    float maximum[FATTN_NUM_LANES][FATTN_TILE_TOKENS]
                 [FATTN_KV_SLOTS_PER_LANE][FATTN_KV_REP],
    float denominator[FATTN_NUM_LANES][FATTN_TILE_TOKENS]
                     [FATTN_KV_SLOTS_PER_LANE][FATTN_KV_REP],
    int lane, int row, int slot, int position, Score score,
    Scale &scale, Denominator &final_denominator,
    bool final_row) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable = score.values complete dim = 1
  for (int rep = 0; rep < FATTN_KV_REP; ++rep) {
#pragma HLS UNROLL
    const float old_max =
        position == 0 ? -FLT_MAX : maximum[lane][row][slot][rep];
    const float old_den =
        position == 0 ? 0.0f : denominator[lane][row][slot][rep];
    if (position == 0) {
      maximum[lane][row][slot][rep] = score.values[rep];
      scale_assign(scale, rep, 0.0f, 1.0f, true);
    } else if (score.values[rep] > old_max) {
      maximum[lane][row][slot][rep] = score.values[rep];
      scale_assign(
          scale, rep, exp_softmax(old_max - score.values[rep]),
          1.0f, true);
    } else {
      maximum[lane][row][slot][rep] = old_max;
      scale_assign(
          scale, rep, 1.0f,
          exp_softmax(score.values[rep] - old_max), false);
    }
    denominator[lane][row][slot][rep] =
        old_den * scale_previous(scale, rep) +
        scale_current(scale, rep);
    if (final_row) {
      denominator_assign(
          final_denominator, rep, denominator[lane][row][slot][rep]);
    }
  }
}
inline void online_sm(
    hls::stream<Score> scores[FATTN_NUM_LANES],
    hls::stream<Scale> scales[FATTN_NUM_LANES],
    hls::stream<Denominator> denominators[FATTN_NUM_LANES],
    int batches, int sequences, int context_start) {
#pragma HLS INLINE off
  float maximum[FATTN_NUM_LANES][FATTN_TILE_TOKENS]
               [FATTN_KV_SLOTS_PER_LANE][FATTN_KV_REP];
  float denominator[FATTN_NUM_LANES][FATTN_TILE_TOKENS]
                   [FATTN_KV_SLOTS_PER_LANE][FATTN_KV_REP];
#pragma HLS ARRAY_PARTITION variable = maximum complete dim = 1
#pragma HLS ARRAY_PARTITION variable = maximum complete dim = 4
#pragma HLS ARRAY_PARTITION variable = denominator complete dim = 1
#pragma HLS ARRAY_PARTITION variable = denominator complete dim = 4
#pragma HLS BIND_STORAGE variable = maximum type = ram_2p impl = bram
#pragma HLS BIND_STORAGE variable = denominator type = ram_2p impl = bram
  const int visible_rows =
      sequences * context_start + sequences * (sequences + 1) / 2;
  const int total_actions =
      batches * visible_rows * FATTN_KV_SLOTS_PER_LANE * FATTN_NUM_LANES;
  int position = 0, slot = 0, row = 0, lane = 0;
  for (int action = 0; action < total_actions; ++action) {
#pragma HLS DEPENDENCE variable = maximum inter false
#pragma HLS DEPENDENCE variable = denominator inter false
#pragma HLS PIPELINE II = 3
    const Score score = scores[lane].read();
    Scale scale;
#pragma HLS ARRAY_PARTITION variable = score.values complete dim = 1
#pragma HLS ARRAY_PARTITION variable = scale.previous complete dim = 1
#pragma HLS ARRAY_PARTITION variable = scale.current complete dim = 1
#pragma HLS ARRAY_PARTITION variable = scale.max_grew complete dim = 1
    Denominator result;
    const bool final_row = position == context_start + row;
    online_sm_step(
        maximum, denominator, lane, row, slot, position, score, scale,
        result, final_row);
    scales[lane].write(scale);
    if (final_row) {
#pragma HLS ARRAY_PARTITION variable = result.values complete dim = 1
      denominators[lane].write(result);
    }
    if (++lane == FATTN_NUM_LANES) {
      lane = 0;
      if (++row == sequences) {
        if (++slot == FATTN_KV_SLOTS_PER_LANE) {
          slot = 0;
          if (++position == context_start + sequences) position = 0;
        }
        row = position < context_start ? 0 : position - context_start;
      }
    }
  }
}
inline void sv_lane(
    hls::stream<Scale> &scales, hls::stream<VChunk> &values,
    hls::stream<AccChunk> &results, int batches, int sequences,
    int start) {
#pragma HLS INLINE off
  float state[FATTN_TILE_TOKENS][FATTN_KV_SLOTS_PER_LANE]
             [FATTN_KV_REP][FATTN_HEAD_DIM];
  Scale row_scales[FATTN_TILE_TOKENS];
#pragma HLS ARRAY_PARTITION variable = state complete dim = 3
#pragma HLS ARRAY_PARTITION variable = state cyclic factor = FATTN_VALUE_LANES dim = 4
#pragma HLS ARRAY_PARTITION variable = row_scales complete dim = 1
#pragma HLS BIND_STORAGE variable = state type = ram_t2p impl = bram
  const int visible_rows =
      sequences * start + sequences * (sequences + 1) / 2;
  const int total_actions =
      batches * visible_rows * FATTN_KV_SLOTS_PER_LANE *
      FATTN_VALUE_CHUNKS;
  int batch = 0, position = 0, slot = 0, chunk = 0, row = 0;
  VChunk value;
sv_update_loop:
  for (int action = 0; action < total_actions; ++action) {
#pragma HLS DEPENDENCE variable = state inter false
#pragma HLS PIPELINE II = 1
    const int first_row = position < start ? 0 : position - start;
    if (row == first_row) value = values.read();
#pragma HLS ARRAY_PARTITION variable = value.values complete dim = 1
    if (chunk == 0) row_scales[row] = scales.read();
    const Scale scale = row_scales[row];
    const bool emit_result =
        position >= start && row == position - start;
    AccChunk result;
#pragma HLS ARRAY_PARTITION variable = scale.previous complete dim = 1
#pragma HLS ARRAY_PARTITION variable = scale.current complete dim = 1
#pragma HLS ARRAY_PARTITION variable = scale.max_grew complete dim = 1
#pragma HLS ARRAY_PARTITION variable = result.values complete dim = 0
    for (int lane = 0; lane < FATTN_VALUE_LANES; ++lane) {
#pragma HLS UNROLL
      const int dimension = chunk * FATTN_VALUE_LANES + lane;
      const float current =
          bf16_to_float(value.values[lane]);
      for (int rep = 0; rep < FATTN_KV_REP; ++rep) {
#pragma HLS UNROLL
        float next;
        if (position == 0) {
          next = current;
        } else {
          const float previous =
              state[row][slot][rep][dimension];
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
    if (emit_result) results.write(result);
    if (++row == sequences) {
      row = first_row;
      if (++chunk == FATTN_VALUE_CHUNKS) {
        chunk = 0;
        if (++slot == FATTN_KV_SLOTS_PER_LANE) {
          slot = 0;
          if (++position == start + sequences) {
            position = 0;
            ++batch;
          }
        }
        row = position < start ? 0 : position - start;
      }
    }
  }
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
