#pragma once

#include "brisk_types.hpp"

namespace brisk {

using query_stream = hls::stream<QPair>;
using key_stream = hls::stream<KChunk>;
using value_stream = hls::stream<VChunk>;
using score_stream = hls::stream<Score>;
using scale_stream = hls::stream<Scale>;
using denominator_stream = hls::stream<Denominator>;
using output_stream = hls::stream<qo_pack_t>;
using incoming_stream = hls::stream<cache_line_t>;

}  // namespace brisk
