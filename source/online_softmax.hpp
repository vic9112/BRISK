//********************************************************
//* Project         : FPGenius - FPT'26 Design Competition Track B
//* Filename        : online_softmax.hpp
//* Author          : VicChen (KUAN-HSI, CHEN)
//* Created On      : 2026/07/10
//* Function        : Implements the online softmax recurrence and final normalization.
//********************************************************

#pragma once

#include "qk_datapath.hpp"
struct Scale {
  float previous[FATTN_KV_REP];
  float current[FATTN_KV_REP];
  bool max_grew[FATTN_KV_REP];
};
inline ap_uint<32> scale_float_bits(float value) {
#pragma HLS INLINE
  union {
    float value;
    std::uint32_t bits;
  } conversion;
  conversion.value = value;
  return ap_uint<32>(conversion.bits);
}
inline float scale_bits_float(ap_uint<32> bits) {
#pragma HLS INLINE
  union {
    std::uint32_t bits;
    float value;
  } conversion;
  conversion.bits = bits.to_uint();
  return conversion.value;
}
inline bool scale_max_grew(const Scale &scale, int rep) {
#pragma HLS INLINE
  return scale.max_grew[rep];
}
inline float scale_nontrivial(const Scale &scale, int rep) {
#pragma HLS INLINE
  return scale.max_grew[rep] ? scale.previous[rep] : scale.current[rep];
}
inline void scale_assign(Scale &scale, int rep, float previous,
                               float current, bool max_grew) {
#pragma HLS INLINE
  scale.previous[rep] = previous;
  scale.current[rep] = current;
  scale.max_grew[rep] = max_grew;
}
inline float scale_previous(const Scale &scale, int rep) {
#pragma HLS INLINE
  return scale.previous[rep];
}
inline float scale_current(const Scale &scale, int rep) {
#pragma HLS INLINE
  return scale.current[rep];
}
inline float scale_denominator_update(float old_den, float factor,
                                             bool max_grew) {
#pragma HLS INLINE
  const float previous = max_grew ? factor : 1.0f;
  const float current = max_grew ? 1.0f : factor;
  return old_den * previous + current;
}
inline float scale_sv_update(float previous, float current,
                                    float factor, bool max_grew) {
#pragma HLS INLINE
  const float multiply = max_grew ? previous : current;
  const float add = max_grew ? current : previous;
  return multiply * factor + add;
}
struct Denominator {
  float values[FATTN_KV_REP];
};
inline void denominator_assign(Denominator &denominator, int rep,
                                     float value) {
#pragma HLS INLINE
  denominator.values[rep] = value;
}
inline float denominator_value(const Denominator &denominator,
                                     int rep) {
#pragma HLS INLINE
  return denominator.values[rep];
}
struct AccChunk {
  float values[FATTN_KV_REP][FATTN_VALUE_LANES];
};
struct AccHalfbeat {
  AccChunk halves[FATTN_HALFBEAT_CHUNKS];
};
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
inline void normalize(
    hls::stream<AccChunk> &accumulators,
    hls::stream<Denominator> &denominators,
    hls::stream<qo_pack_t> &output, int actions) {
#pragma HLS INLINE off
  float lower[FATTN_KV_REP][FATTN_QO_HEAD_PACKS][FATTN_VALUE_LANES];
  float upper[FATTN_KV_REP][FATTN_QO_HEAD_PACKS][FATTN_VALUE_LANES];
#pragma HLS ARRAY_PARTITION variable = lower complete dim = 1
#pragma HLS ARRAY_PARTITION variable = lower complete dim = 3
#pragma HLS ARRAY_PARTITION variable = upper complete dim = 1
#pragma HLS ARRAY_PARTITION variable = upper complete dim = 3
  Denominator denominator;
  int chunk = 0, output_word = 0;
  bool output_phase = false;
  for (int action = 0; action < actions; ++action) {
#pragma HLS PIPELINE II = 1
    if (!output_phase) {
      const AccChunk packet = accumulators.read();
#pragma HLS ARRAY_PARTITION variable = packet.values complete dim = 0
      const int pack = chunk >> 1;
      for (int rep = 0; rep < FATTN_KV_REP; ++rep) {
#pragma HLS UNROLL
        for (int lane = 0; lane < FATTN_VALUE_LANES; ++lane) {
#pragma HLS UNROLL
          if ((chunk & 1) == 0)
            lower[rep][pack][lane] = packet.values[rep][lane];
          else
            upper[rep][pack][lane] = packet.values[rep][lane];
        }
      }
      if (++chunk == FATTN_VALUE_CHUNKS) {
        denominator = denominators.read();
        output_phase = true;
        output_word = 0;
      }
    } else {
      const int rep = output_word / FATTN_QO_HEAD_PACKS;
      const int pack = output_word % FATTN_QO_HEAD_PACKS;
      const float inverse = 1.0f / denominator_value(denominator, rep);
      bf16_t lower_bf16[FATTN_VALUE_LANES];
      bf16_t upper_bf16[FATTN_VALUE_LANES];
#pragma HLS ARRAY_PARTITION variable = lower_bf16 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = upper_bf16 complete dim = 1
      for (int lane = 0; lane < FATTN_VALUE_LANES; ++lane) {
#pragma HLS UNROLL
        lower_bf16[lane] =
            float_to_bf16(lower[rep][pack][lane] * inverse);
        upper_bf16[lane] =
            float_to_bf16(upper[rep][pack][lane] * inverse);
      }
      output.write(pack_output(pack_bf16x8(lower_bf16),
                                     pack_bf16x8(upper_bf16)));
      if (++output_word == FATTN_KV_REP * FATTN_QO_HEAD_PACKS) {
        output_phase = false;
        chunk = 0;
      }
    }
  }
}
