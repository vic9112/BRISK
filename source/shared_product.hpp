//********************************************************
//* Project         : FPGenius - FPT'26 Design Competition Track B
//* Filename        : shared_product.hpp
//* Author          : VicChen (KUAN-HSI, CHEN)
//* Created On      : 2026/07/10
//* Function        : Implements the shared floating-product identities used by the datapath.
//********************************************************

#pragma once

#include "attention.hpp"  // Alternate product-bank helpers remain unchanged; active QK uses canonical products.
constexpr int FATTN_FIP_PAIRS_PER_CHUNK = FATTN_VALUE_LANES / 2;
constexpr int FATTN_HYBRID_CLASSICAL_REPS =
    FATTN_KV_REP - FATTN_HYBRID_FIP_REPS;
constexpr int FATTN_FIP_MAIN_PRODUCTS =
    FATTN_HYBRID_FIP_REPS * FATTN_FIP_PAIRS_PER_CHUNK;
constexpr int FATTN_FIP_BETA_PRODUCTS = FATTN_FIP_PAIRS_PER_CHUNK;
constexpr int FATTN_CLASSICAL_PRODUCTS =
    FATTN_HYBRID_CLASSICAL_REPS * FATTN_VALUE_LANES;
constexpr int FATTN_CLASSICAL_PRODUCT_BASE =
    FATTN_FIP_MAIN_PRODUCTS + FATTN_FIP_BETA_PRODUCTS;
constexpr int FATTN_FIP_MEDDSP_PRODUCTS = 7;
constexpr int FATTN_HYBRID_KEY_PRODUCTS =
    FATTN_CLASSICAL_PRODUCT_BASE + FATTN_CLASSICAL_PRODUCTS;
constexpr int FATTN_HYBRID_PRODUCT_BANK = FATTN_HYBRID_KEY_PRODUCTS;
static_assert(FATTN_HYBRID_KEY_PRODUCTS == 28,
              "hybrid GQA key action must use twenty-eight products");
static_assert(FATTN_HYBRID_PRODUCT_BANK == 28,
              "softmax-quotient FIP must use twenty-eight physical products");
static_assert(FATTN_FIP_MEDDSP_PRODUCTS < FATTN_CLASSICAL_PRODUCT_BASE,
              "the seven-product medium-DSP bank must stay inside FIP/beta");
struct HybridOperands {
  float left[FATTN_HYBRID_PRODUCT_BANK];
  float right[FATTN_HYBRID_PRODUCT_BANK];
};
struct HybridProducts {
  float values[FATTN_HYBRID_PRODUCT_BANK];
};
constexpr int FATTN_TEMPORAL_ROW_PRODUCTS =
    FATTN_FIP_MAIN_PRODUCTS + FATTN_CLASSICAL_PRODUCTS;
constexpr int FATTN_TEMPORAL_PHYSICAL_PRODUCTS =
    FATTN_TEMPORAL_ROW_PRODUCTS + FATTN_FIP_BETA_PRODUCTS;
static_assert(FATTN_TEMPORAL_ROW_PRODUCTS == 24,
              "temporal beta reuse keeps twenty-four row products");
static_assert(FATTN_TEMPORAL_PHYSICAL_PRODUCTS == 28,
              "first-row beta plus row products must retain the selected bank");
struct TemporalRowProducts {
  float main[FATTN_FIP_MAIN_PRODUCTS];
  float classical[FATTN_CLASSICAL_PRODUCTS];
};
inline void hybrid_product_bank(const HybridOperands &operands,
                                HybridProducts &products) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable = operands.left complete dim = 1
#pragma HLS ARRAY_PARTITION variable = operands.right complete dim = 1
#pragma HLS ARRAY_PARTITION variable = products.values complete dim = 1
  for (int lane = 0; lane < FATTN_FIP_MEDDSP_PRODUCTS; ++lane) {
#pragma HLS UNROLL
    float product;
#pragma HLS BIND_OP variable = product op = fmul impl = meddsp
    product = operands.left[lane] * operands.right[lane];
    products.values[lane] = product;
  }
  for (int lane = FATTN_FIP_MEDDSP_PRODUCTS;
       lane < FATTN_HYBRID_PRODUCT_BANK; ++lane) {
#pragma HLS UNROLL
    products.values[lane] = operands.left[lane] * operands.right[lane];
  }
}
inline void temporal_row_product_bank(
    const HybridOperands &operands,
    TemporalRowProducts &products) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable = operands.left complete dim = 1
#pragma HLS ARRAY_PARTITION variable = operands.right complete dim = 1
#pragma HLS ARRAY_PARTITION variable = products.main complete dim = 1
#pragma HLS ARRAY_PARTITION variable = products.classical complete dim = 1
  for (int lane = 0; lane < FATTN_FIP_MEDDSP_PRODUCTS; ++lane) {
#pragma HLS UNROLL
    float product;
#pragma HLS BIND_OP variable = product op = fmul impl = meddsp
    product = operands.left[lane] * operands.right[lane];
    products.main[lane] = product;
  }
  products.main[FATTN_FIP_MEDDSP_PRODUCTS] =
      operands.left[FATTN_FIP_MEDDSP_PRODUCTS] *
      operands.right[FATTN_FIP_MEDDSP_PRODUCTS];
  for (int lane = 0; lane < FATTN_CLASSICAL_PRODUCTS; ++lane) {
#pragma HLS UNROLL
    products.classical[lane] =
        operands.left[FATTN_CLASSICAL_PRODUCT_BASE + lane] *
        operands.right[FATTN_CLASSICAL_PRODUCT_BASE + lane];
  }
}
