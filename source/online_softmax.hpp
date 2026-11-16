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
inline ap_uint<32> scale_float_bits(float value) {  // Retained bit-conversion helper.
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
inline float scale_denominator_update(float old_den, float factor,  // Retained helper; online_sm_step keeps its own expression.
                                      bool max_grew) {
#pragma HLS INLINE
  const float previous = max_grew ? factor : 1.0f;
  const float current = max_grew ? 1.0f : factor;
  return old_den * previous + current;
}
inline float scale_sv_update(float previous, float current,  // Retained helper; selected SV is sv_lane.
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
// Preserve HLS source-location identifiers after separating alternatives.
#line 155
inline void normalize(  // Collect a completed row, then share each reciprocal across its packed dimensions.
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
      const float inverse = 1.0f / denominator_value(denominator, rep);  // Share this reciprocal across the packed word.
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
      output.write(pack_output(pack_bf16x8(lower_bf16),  // Retain reciprocal-then-multiply rounding and full-word service.
                                     pack_bf16x8(upper_bf16)));
      if (++output_word == FATTN_KV_REP * FATTN_QO_HEAD_PACKS) {
        output_phase = false;
        chunk = 0;
      }
    }
  }
}

// Retained alternatives are not selected by the active kernel.
#include "softmax_alternatives.hpp"
