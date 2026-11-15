#pragma once

#include "attention.hpp"

namespace brisk {

inline void normalize_scores(score_stream scores[lanes],
                             scale_stream scales[lanes],
                             denominator_stream denominators[lanes],
                             const TileRequest& request) {
#pragma HLS INLINE off
    // Batch is the outer traversal, so its state can reuse these tile banks.
    float maximum[lanes][tile_rows][local_kv_heads][representatives];
    float sum[lanes][tile_rows][local_kv_heads][representatives];
#pragma HLS ARRAY_PARTITION variable=maximum complete dim=4
#pragma HLS ARRAY_PARTITION variable=sum complete dim=4
#pragma HLS BIND_STORAGE variable=maximum type=ram_t2p impl=bram
#pragma HLS BIND_STORAGE variable=sum type=ram_t2p impl=bram
    for (int batch = 0; batch < request.batches; ++batch) {
        for (int position = 0;
             position < request.context_start + request.sequences; ++position) {
            for (int head = 0; head < local_kv_heads; ++head) {
                for (int row = first_visible_row(position, request.context_start);
                     row < request.sequences; ++row) {
                    for (int lane = 0; lane < lanes; ++lane) {
#pragma HLS PIPELINE II=1 style=flp
                        Score score;
                        switch (lane) {
                            case 0: score = scores[0].read(); break;
                            case 1: score = scores[1].read(); break;
                            case 2: score = scores[2].read(); break;
                            default: score = scores[3].read(); break;
                        }
                        Scale scale;
                        Denominator denominator;
                        for (int rep = 0; rep < representatives; ++rep) {
#pragma HLS UNROLL
                            const float x = score.values[rep];
                            float next_maximum;
                            float next_sum;
                            if (position == 0) {
                                next_maximum = x;
                                next_sum = 1.0f;
                                scale.previous[rep] = 0.0f;
                                scale.current[rep] = 1.0f;
                                scale.max_grew[rep] = true;
                            } else {
                                const float old_maximum = maximum[lane][row][head][rep];
                                const bool grew = x > old_maximum;
                                next_maximum = grew ? x : old_maximum;
                                const float previous = grew ? std::exp(old_maximum - x) : 1.0f;
                                const float current = grew ? 1.0f : std::exp(x - old_maximum);
                                next_sum = sum[lane][row][head][rep] * previous + current;
                                scale.previous[rep] = previous;
                                scale.current[rep] = current;
                                scale.max_grew[rep] = grew;
                            }
                            maximum[lane][row][head][rep] = next_maximum;
                            sum[lane][row][head][rep] = next_sum;
                            denominator.values[rep] = next_sum;
                        }
                        const bool final = position == request.context_start + row;
                        switch (lane) {
                            case 0: scales[0].write(scale);
                                    if (final) denominators[0].write(denominator); break;
                            case 1: scales[1].write(scale);
                                    if (final) denominators[1].write(denominator); break;
                            case 2: scales[2].write(scale);
                                    if (final) denominators[2].write(denominator); break;
                            default: scales[3].write(scale);
                                     if (final) denominators[3].write(denominator); break;
                        }
                    }
                }
            }
        }
    }
}

}  // namespace brisk
