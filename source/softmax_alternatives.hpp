// Unselected interleaved softmax, retained for source compatibility.
// The kernel uses online_sm in kv_stream.hpp instead of this function.
#pragma once

#include "online_softmax.hpp"

inline void stream_softmax_interleaved(
    hls::stream<Score> scores[FATTN_NUM_LANES],
    hls::stream<Scale> scales[FATTN_NUM_LANES],
    hls::stream<Denominator> denominators[FATTN_NUM_LANES],
    int sequence_count, int context_start, int actions_per_lane) {
#pragma HLS INLINE off
  float maximum[FATTN_NUM_LANES][FATTN_KV_REP];
  float denominator[FATTN_NUM_LANES][FATTN_KV_REP];
#pragma HLS ARRAY_PARTITION variable = maximum complete dim = 2
#pragma HLS ARRAY_PARTITION variable = denominator complete dim = 2
  int sequence = 0, slot = 0, position = 0;
  for (int action = 0; action < (actions_per_lane << 2); ++action) {
#pragma HLS PIPELINE II = 3
    const int lane = action & (FATTN_NUM_LANES - 1);
    const int visible = context_start + sequence + 1;
    if (position < visible) {
      const Score score = scores[lane].read();
      Scale scale;
#pragma HLS ARRAY_PARTITION variable = score.values complete dim = 1
#pragma HLS ARRAY_PARTITION variable = scale.previous complete dim = 1
#pragma HLS ARRAY_PARTITION variable = scale.current complete dim = 1
#pragma HLS ARRAY_PARTITION variable = scale.max_grew complete dim = 1
      for (int rep = 0; rep < FATTN_KV_REP; ++rep) {
#pragma HLS UNROLL
        const float old_max = position == 0 ? -FLT_MAX : maximum[lane][rep];
        const float old_den = position == 0 ? 0.0f : denominator[lane][rep];
        if (old_max == -FLT_MAX) {
          maximum[lane][rep] = score.values[rep];
          scale_assign(scale, rep, 0.0f, 1.0f, true);
        } else if (score.values[rep] > old_max) {
          maximum[lane][rep] = score.values[rep];
          scale_assign(
              scale, rep, exp_softmax(old_max - score.values[rep]),
              1.0f, true);
        } else {
          maximum[lane][rep] = old_max;
          scale_assign(
              scale, rep, 1.0f,
              exp_softmax(score.values[rep] - old_max), false);
        }
        denominator[lane][rep] =
            old_den * scale_previous(scale, rep) +
            scale_current(scale, rep);
      }
      scales[lane].write(scale);
      if (lane == FATTN_NUM_LANES - 1) ++position;
    } else {
      Denominator result;
#pragma HLS ARRAY_PARTITION variable = result.values complete dim = 1
      for (int rep = 0; rep < FATTN_KV_REP; ++rep) {
#pragma HLS UNROLL
        denominator_assign(result, rep, denominator[lane][rep]);
      }
      denominators[lane].write(result);
      if (lane == FATTN_NUM_LANES - 1) {
        position = 0;
        if (++slot == FATTN_KV_SLOTS_PER_LANE) {
          slot = 0;
          if (++sequence == sequence_count) sequence = 0;
        }
      }
    }
  }
}
