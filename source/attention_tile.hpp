//********************************************************
//* Project         : FPGenius - FPT'26 Design Competition Track B
//* Filename        : attention_tile.hpp
//* Author          : VicChen (KUAN-HSI, CHEN)
//* Created On      : 2026/07/10
//* Function        : Defines the bounded row state shared by the streaming attention stages.
//********************************************************

#pragma once

#include "online_softmax.hpp"
struct VChunk {
  bf16_t values[FATTN_VALUE_LANES];
};
struct VHalfbeat {
  VChunk halves[FATTN_HALFBEAT_CHUNKS];
};
using DimensionKPacket = KChunk;
using DimensionVPacket = VChunk;
using DimensionAccPacket = AccChunk;
inline float sv_classical_update(float previous, float previous_factor,
                                       float current_weight, float value,
                                       bool has_previous) {
#pragma HLS INLINE
  const float carried =
      has_previous ? previous * previous_factor : 0.0f;
  return carried + current_weight * value;
}
inline float sv_classical_pair_update(
    float previous, float previous_factor, float weight0, float weight1,
    float value0, float value1, bool has_previous) {
#pragma HLS INLINE
  const float first = sv_classical_update(
      previous, previous_factor, weight0, value0, has_previous);
  return first + weight1 * value1;
}
