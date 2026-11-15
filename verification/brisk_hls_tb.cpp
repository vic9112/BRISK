// Independent DV: derived only from the admitted specification/public types.
// No DUT helper, implementation source, or candidate output supplies an oracle.
#include "../spec/brisk_types.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kLanes = 4;
constexpr int kQueryHeads = 32;
constexpr int kKvHeads = 8;
constexpr int kDimension = 128;
constexpr int kGuardWords = 4;
// Public maximum extents also bound Vitis 2025.2 wrapc's unconditional TV copies.
constexpr std::size_t kTransportQoWords = 1048576;
constexpr std::size_t kTransportActiveWords = 2097152;
constexpr std::size_t kTransportBackingLines = 262144;
constexpr std::uint16_t kSentinel = 0xc2c8;  // Finite BF16 -100.
int successful_calls = 0;
int rejected_calls = 0;
std::uint64_t compared_elements = 0;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

float expand(std::uint16_t value) {
    std::uint32_t bits = std::uint32_t(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

std::uint16_t round_bf16(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffU + ((bits >> 16) & 1U);
    return static_cast<std::uint16_t>(bits >> 16);
}

float from_bits(std::uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

bool accepted(float actual, float reference) {
    if (!std::isfinite(actual) || !std::isfinite(reference)) return false;
    const float error = std::fabs(actual - reference);
    return error <= std::max(0.0625f,
        0.08f * std::max(std::fabs(reference), 1.0e-6f));
}

template <int Width>
void set_element(ap_uint<Width>& word, int element, std::uint16_t value) {
    word.range(16 * element + 15, 16 * element) = value;
}

template <int Width>
std::uint16_t element(const ap_uint<Width>& word, int index) {
    return static_cast<std::uint16_t>(word.range(16 * index + 15,
                                               16 * index).to_uint());
}

template <int Width>
ap_uint<Width> sentinel_word() {
    ap_uint<Width> result = 0;
    for (int i = 0; i < Width / 16; ++i) set_element(result, i, kSentinel);
    return result;
}

std::uint32_t mix(std::uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    return x ^ (x >> 16);
}

enum class Pattern { Asymmetric, Uniform, One, Ties, ScorePrecision, ProbabilityPrecision };

std::uint16_t sample(Pattern pattern, int seed, int kind, int batch,
                     int position, int head, int dimension) {
    if (pattern == Pattern::ScorePrecision || pattern == Pattern::ProbabilityPrecision) {
        if (kind == 0) return dimension == 0 ? 0x4300U : 0;  // Q[0] = 128.
        if (kind == 1) return dimension == 0
            ? static_cast<std::uint16_t>(position == 0 ? 0x4110U : 0x4111U) : 0;
        if (pattern == Pattern::ScorePrecision)
            return position == 0 ? 0xbf80U : 0x3f80U;
        return position == 0 ? 0xc300U : 0x427cU;  // V = -128, +63.
    }
    if (pattern != Pattern::Asymmetric && kind != 2) return 0;
    if (pattern == Pattern::One) return 0x3f80;
    if (pattern == Pattern::Ties) {
        const unsigned base = (dimension % 2 == 0) ? 0x3f80U : 0x3f81U;
        return static_cast<std::uint16_t>(base + (position != 0 ? 1U : 0U)
            + (dimension % 4 >= 2 ? 0x8000U : 0U));
    }
    if (pattern == Pattern::Uniform) {
        const float value = (position % 17 - 8) * 0.25f
            + (batch * 8 + head) * 0.5f + (dimension % 8) * 0.0625f;
        return round_bf16(value);
    }
    const std::uint32_t code = mix(std::uint32_t(seed) * 0x9e3779b9U
        + std::uint32_t(kind + 1) * 0x8da6b343U
        + std::uint32_t(batch + 1) * 0xd8163841U
        + std::uint32_t(position + 1) * 0xcb1ab31fU
        + std::uint32_t(head + 1) * 0x165667b1U
        + std::uint32_t(dimension + 1) * 0x27d4eb2dU);
    if (kind == 2) {
        return round_bf16((int(code % 65U) - 32) * 0.125f
            + batch * 4.0f + head * 0.75f + seed * 0.03125f);
    }
    return round_bf16((int(code % 33U) - 16) * 0.0625f);
}

struct Controls {
    int batches;
    int sequences;
    int position;
    int stride;
    int valid;
    int mask;
    int flags;
    int identity;
};

int expected_status(const Controls& c) {
    const std::int64_t end = std::int64_t(c.position) + c.sequences;
    if (c.batches < 1 || c.batches > 2 || c.sequences < 1
        || c.position < 0 || c.stride > 8192 || end > c.stride)
        return 1;
    if (c.valid != c.position) return 2;
    if ((c.sequences != 1 && c.mask != 1)
        || (c.sequences == 1 && c.mask != 0 && c.mask != 1))
        return 3;
    if (c.flags != 0) return 4;
    return 0;
}

struct CallBuffers {
    std::array<std::vector<qo_pack_t>, kLanes> query;
    std::array<std::vector<qo_pack_t>, kLanes> output;
    std::vector<kv_pair_pack_t> active;
    std::size_t output_words;

    CallBuffers(int batches, int sequences)
        : active(std::size_t(batches) * sequences * 128 + kGuardWords,
                 sentinel_word<256>()),
          output_words(std::size_t(batches) * sequences * 64) {
        for (int lane = 0; lane < kLanes; ++lane) {
            query[lane].assign(output_words + kGuardWords, sentinel_word<256>());
            output[lane].assign(output_words + kGuardWords, sentinel_word<256>());
        }
    }
};

using Backing = std::array<std::vector<cache_line_t>, kLanes>;

template <int Width>
void check_buffer(const std::vector<ap_uint<Width>>& actual,
                  const std::vector<ap_uint<Width>>& expected,
                  const std::string& label) {
    require(actual.size() >= expected.size(), label + ": insufficient transport extent");
    const auto sentinel = sentinel_word<Width>();
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const auto value = index < expected.size() ? expected[index] : sentinel;
        if (actual[index] != value)
            throw std::runtime_error(label + ": word=" + std::to_string(index));
    }
}

struct Transport {
    std::array<std::vector<qo_pack_t>, kLanes> query;
    std::array<std::vector<qo_pack_t>, kLanes> output;
    std::vector<kv_pair_pack_t> active;
    std::size_t prior_qo_words = 0;
    std::size_t prior_active_words = 0;

    Transport() : active(kTransportActiveWords + kGuardWords, sentinel_word<256>()) {
        for (int lane = 0; lane < kLanes; ++lane) {
            query[lane].assign(kTransportQoWords + kGuardWords, sentinel_word<256>());
            output[lane].assign(kTransportQoWords + kGuardWords, sentinel_word<256>());
        }
    }

    void load(const CallBuffers& call) {
        require(call.query[0].size() <= query[0].size()
                && call.active.size() <= active.size(), "transport capacity");
        for (int lane = 0; lane < kLanes; ++lane) {
            std::fill_n(query[lane].begin(), prior_qo_words, sentinel_word<256>());
            std::fill_n(output[lane].begin(), prior_qo_words, sentinel_word<256>());
            std::copy(call.query[lane].begin(), call.query[lane].end(), query[lane].begin());
            std::copy(call.output[lane].begin(), call.output[lane].end(), output[lane].begin());
        }
        std::fill_n(active.begin(), prior_active_words, sentinel_word<256>());
        std::copy(call.active.begin(), call.active.end(), active.begin());
        prior_qo_words = call.query[0].size();
        prior_active_words = call.active.size();
    }

    void collect(CallBuffers& call) const {
        const auto sentinel = sentinel_word<256>();
        for (int lane = 0; lane < kLanes; ++lane) {
            check_buffer(query[lane], call.query[lane], "query transport lane="
                         + std::to_string(lane));
            for (std::size_t i = call.output[lane].size(); i < output[lane].size(); ++i) {
                if (output[lane][i] != sentinel)
                    throw std::runtime_error("output transport padding lane="
                        + std::to_string(lane) + " word=" + std::to_string(i));
            }
            std::copy_n(output[lane].begin(), call.output[lane].size(), call.output[lane].begin());
        }
        check_buffer(active, call.active, "active transport");
    }
};

void invoke(CallBuffers& call, Backing& backing, const Controls& c,
            int& status) {
    // Q/active/O are invocation-local, so reuse their transport allocation.
    // Backing is never pooled: each live Session retains disjoint fixed pointers.
    static Transport transport;
    transport.load(call);
    llama3_fast_attention(
        transport.query[0].data(), transport.query[1].data(),
        transport.query[2].data(), transport.query[3].data(), transport.active.data(),
        backing[0].data(), backing[1].data(),
        backing[2].data(), backing[3].data(),
        transport.output[0].data(), transport.output[1].data(),
        transport.output[2].data(), transport.output[3].data(),
        c.batches, c.sequences, c.position, c.stride, c.valid,
        c.mask, c.flags, c.identity, &status);
    transport.collect(call);
}

void compare_backing(const Backing& actual, const Backing& expected,
                     const std::string& label) {
    for (int lane = 0; lane < kLanes; ++lane) {
        check_buffer(actual[lane], expected[lane], label + ": backing lane="
                     + std::to_string(lane));
    }
}

struct Session {
    const std::string name;
    const int batches;
    const int stride;
    const int identity;
    const Pattern pattern;
    const int seed;
    int end;
    Backing backing;
    Backing expected_backing;
    std::vector<std::uint16_t> keys;
    std::vector<std::uint16_t> values;
    std::array<std::array<int, 40>, 2> resident;

    Session(std::string label, int batch_count, int cache_stride,
            int cache_identity, Pattern data_pattern, int data_seed,
            int cold_history = 0)
        : name(std::move(label)), batches(batch_count), stride(cache_stride),
          identity(cache_identity), pattern(data_pattern), seed(data_seed),
          end(cold_history),
          keys(std::size_t(batch_count) * cache_stride * kKvHeads * kDimension),
          values(keys.size()) {
        for (auto& slots : resident) slots.fill(-1);
        for (int lane = 0; lane < kLanes; ++lane) {
            backing[lane].assign(kTransportBackingLines + kGuardWords, sentinel_word<512>());
            expected_backing[lane].assign(std::size_t(batches) * stride * 16 + kGuardWords,
                                          sentinel_word<512>());
        }
        for (int p = 0; p < cold_history; ++p) fill_record(p);
        for (int b = 0; b < batches; ++b)
            for (int p = 0; p < cold_history; ++p) store_record(b, p);
        for (int lane = 0; lane < kLanes; ++lane)
            std::copy(expected_backing[lane].begin(), expected_backing[lane].end(),
                      backing[lane].begin());
    }

    std::size_t index(int b, int p, int h, int d) const {
        return (((std::size_t(b) * stride + p) * kKvHeads + h) * kDimension + d);
    }

    void fill_record(int p) {
        for (int b = 0; b < batches; ++b)
            for (int h = 0; h < kKvHeads; ++h)
                for (int d = 0; d < kDimension; ++d) {
                    keys[index(b, p, h, d)] = sample(pattern, seed, 1, b, p, h, d);
                    values[index(b, p, h, d)] = sample(pattern, seed, 2, b, p, h, d);
                }
    }

    void store_record(int b, int p) {
        for (int h = 0; h < kKvHeads; ++h) {
            const int lane = h / 2;
            for (int line = 0; line < 8; ++line) {
                cache_line_t packed = 0;
                for (int pair = 0; pair < 2; ++pair)
                    for (int i = 0; i < 8; ++i) {
                        const int d = line * 16 + pair * 8 + i;
                        set_element(packed, pair * 16 + i, keys[index(b, p, h, d)]);
                        set_element(packed, pair * 16 + 8 + i,
                                    values[index(b, p, h, d)]);
                    }
                const std::size_t address = ((std::size_t(b) * stride + p) * 2
                                               + h % 2) * 8 + line;
                expected_backing[lane][address] = packed;
            }
        }
    }

    void prepare(CallBuffers& call, int sequences) const {
        for (int b = 0; b < batches; ++b)
            for (int r = 0; r < sequences; ++r) {
                const int p = end + r;
                for (int h = 0; h < kQueryHeads; ++h)
                    for (int d = 0; d < kDimension; ++d) {
                        const std::size_t word = ((std::size_t(b) * sequences + r)
                            * 8 + h % 8) * 8 + d / 16;
                        set_element(call.query[h / 8][word], d % 16,
                                    sample(pattern, seed, 0, b, p, h, d));
                    }
                for (int h = 0; h < kKvHeads; ++h)
                    for (int d = 0; d < kDimension; ++d) {
                        const std::size_t word = ((std::size_t(b) * sequences + r)
                            * 8 + h) * 16 + d / 8;
                        set_element(call.active[word], d % 8, keys[index(b, p, h, d)]);
                        set_element(call.active[word], 8 + d % 8,
                                    values[index(b, p, h, d)]);
                    }
            }
    }

    // Offline dense softmax intentionally differs from the DUT's online update.
    // Only BF16 ingress/egress is quantized. Scores/weights/sums stay FP32.
    void check_output(const CallBuffers& call, int sequences,
                      const std::string& label) const {
        const float scale = 1.0f / std::sqrt(float(kDimension));
        for (int b = 0; b < batches; ++b)
            for (int r = 0; r < sequences; ++r)
                for (int h = 0; h < kQueryHeads; ++h) {
                    const int visible = end + r + 1;
                    std::vector<float> weight(visible);
                    float maximum = -std::numeric_limits<float>::infinity();
                    for (int p = 0; p < visible; ++p) {
                        float dot = 0.0f;
                        for (int d = 0; d < kDimension; ++d) {
                            const float q = expand(sample(pattern, seed, 0,
                                                          b, end + r, h, d));
                            const float k = expand(keys[index(b, p, h / 4, d)]);
                            const float product = q * k;
                            dot = dot + product;
                        }
                        weight[p] = dot * scale;
                        maximum = std::max(maximum, weight[p]);
                    }
                    float denominator = 0.0f;
                    for (float& w : weight) {
                        w = std::exp(w - maximum);
                        denominator = denominator + w;
                    }
                    for (int d = 0; d < kDimension; ++d) {
                        float sum = 0.0f;
                        for (int p = 0; p < visible; ++p) {
                            const float term = weight[p]
                                * expand(values[index(b, p, h / 4, d)]);
                            sum = sum + term;
                        }
                        const std::uint16_t reference_bits = round_bf16(sum / denominator);
                        const std::size_t word = ((std::size_t(b) * sequences + r)
                            * 8 + h % 8) * 8 + d / 16;
                        const auto actual_bits = element(call.output[h / 8][word], d % 16);
                        const float actual = expand(actual_bits);
                        const float reference = expand(reference_bits);
                        const bool exact_case = pattern == Pattern::Uniform
                            || pattern == Pattern::One || pattern == Pattern::Ties;
                        const bool match = exact_case
                            ? (std::isfinite(actual) && std::isfinite(reference)
                               && actual_bits == reference_bits)
                            : accepted(actual, reference);
                        if (!match) {
                            std::cerr << "FIRST_DIVERGENCE case=" << label
                                << " lane=" << h / 8 << " batch=" << b
                                << " row=" << r << " head=" << h
                                << " dim=" << d << " actual_bf16=0x" << std::hex
                                << actual_bits << " reference_bf16=0x" << reference_bits
                                << std::dec << " actual=" << actual
                                << " reference=" << reference << '\n';
                            throw std::runtime_error(label + ": output mismatch");
                        }
                        ++compared_elements;
                    }
                }
        for (int lane = 0; lane < kLanes; ++lane)
            for (std::size_t i = call.output_words; i < call.output[lane].size(); ++i)
                require(call.output[lane][i] == sentinel_word<256>(),
                        label + ": output guard changed");
    }

    void append(int sequences, int mask = 1) {
        const std::string label = name + "/P" + std::to_string(end)
            + "/S" + std::to_string(sequences);
        const Controls controls{batches, sequences, end, stride, end, mask, 0, identity};
        require(expected_status(controls) == 0, label + ": illegal TB controls");
        for (int p = end; p < end + sequences; ++p) fill_record(p);
        CallBuffers call(batches, sequences);
        prepare(call, sequences);
        const auto query_before = call.query;
        const auto active_before = call.active;
        const int window = 40 / batches;
        for (int b = 0; b < batches; ++b)
            for (int p = end; p < end + sequences; ++p) {
                int& old_position = resident[b][p % window];
                if (old_position >= 0) store_record(b, old_position);
                old_position = p;
            }
        int status = -777;
        invoke(call, backing, controls, status);
        require(status == 0, label + ": status=" + std::to_string(status));
        require(call.query == query_before && call.active == active_before,
                label + ": input buffer changed");
        compare_backing(backing, expected_backing, label);
        check_output(call, sequences, label);
        end += sequences;
        ++successful_calls;
        std::cout << "CASE_PASS " << label << '\n';
    }
};

void oracle_self_check() {
    require(accepted(1.0625f, 1.0f), "comparator absolute bound");
    require(accepted(1.078125f, 1.0f), "comparator OR relative branch");
    require(!accepted(1.125f, 1.0f), "comparator must reject 1 versus 1.125");
    require(accepted(0.0625f, 0.0f), "comparator zero absolute branch");
    require(!accepted(0.0625001f, 0.0f), "comparator zero rejection");
    require(!accepted(std::numeric_limits<float>::infinity(), 1.0f),
            "comparator infinity rejection");
    require(!accepted(1.0f, std::numeric_limits<float>::quiet_NaN()),
            "comparator NaN rejection");
    require(round_bf16(from_bits(0x3f808000U)) == 0x3f80U, "RNE even lower");
    require(round_bf16(from_bits(0x3f818000U)) == 0x3f82U, "RNE even upper");
    require(round_bf16(from_bits(0xbf808000U)) == 0xbf80U, "RNE negative lower");
    require(round_bf16(from_bits(0xbf818000U)) == 0xbf82U, "RNE negative upper");
    require(round_bf16(-0.0f) == 0x8000U, "RNE signed zero");
    require(expand(0x3f80U) == 1.0f && expand(0xbf80U) == -1.0f,
            "BF16 exact expansion");
    const float scale = 1.0f / std::sqrt(128.0f);
    const float score0 = (128.0f * 9.0f) * scale;
    const float score1 = (128.0f * 9.0625f) * scale;
    auto two_record = [](float s0, float s1, float v0, float v1, bool rounded_probability) {
        const float maximum = std::max(s0, s1);
        const float w0 = std::exp(s0 - maximum);
        const float w1 = std::exp(s1 - maximum);
        const float denominator = w0 + w1;
        if (rounded_probability) {
            const float p0 = expand(round_bf16(w0 / denominator));
            const float p1 = expand(round_bf16(w1 / denominator));
            return expand(round_bf16(p0 * v0 + p1 * v1));
        }
        return expand(round_bf16((w0 * v0 + w1 * v1) / denominator));
    };
    const float score_reference = two_record(score0, score1, -1.0f, 1.0f, false);
    const float rounded_score = two_record(expand(round_bf16(score0)),
        expand(round_bf16(score1)), -1.0f, 1.0f, false);
    require(!accepted(rounded_score, score_reference), "score discriminator sensitivity");
    const float probability_reference = two_record(score0, score1, -128.0f, 63.0f, false);
    const float rounded_probability = two_record(score0, score1, -128.0f, 63.0f, true);
    require(!accepted(rounded_probability, probability_reference),
            "probability discriminator sensitivity");
    const float rounded_factor = expand(round_bf16(std::exp(score0 - score1)));
    const float rounded_factor_result = expand(round_bf16(
        (-128.0f * rounded_factor + 63.0f) / (rounded_factor + 1.0f)));
    require(!accepted(rounded_factor_result, probability_reference),
            "factor discriminator sensitivity");
    std::cout << "ORACLE_SELF_CHECK_PASS\n";
}

void numeric_suite() {
    Session asymmetric("asymmetric-B2", 2, 29, 1, Pattern::Asymmetric, 7);
    asymmetric.append(3);
    asymmetric.append(1, 0);
    asymmetric.append(1, 1);
    Session ones("analytic-one", 1, 7, 2, Pattern::One, 0);
    ones.append(1);
    Session ties("analytic-ties", 1, 7, 3, Pattern::Ties, 0);
    ties.append(2);
    Session scores("FP32-score-discriminator", 1, 7, 12,
                   Pattern::ScorePrecision, 0);
    scores.append(2);
    Session probabilities("FP32-probability-discriminator", 1, 7, 13,
                          Pattern::ProbabilityPrecision, 0);
    probabilities.append(2);
}

void invalid_suite() {
    Session live("invalid-state-preservation", 1, 43, 20, Pattern::Uniform, 0);
    live.append(3);
    const Controls legal{1, 1, live.end, live.stride, live.end, 1, 0, live.identity};
    struct InvalidCase { const char* name; Controls controls; int status; };
    std::vector<InvalidCase> cases;
    auto add = [&](const char* name, Controls controls, int status) {
        require(expected_status(controls) == status, std::string(name) + ": oracle status");
        cases.push_back({name, controls, status});
    };
    Controls c = legal;
    c.batches = 0; add("batch-zero", c, 1);
    c = legal; c.batches = 3; add("batch-three", c, 1);
    c = legal; c.sequences = 0; add("length-zero", c, 1);
    c = legal; c.sequences = -1; add("length-negative", c, 1);
    c = legal; c.position = -1; add("position-negative", c, 1);
    c = legal; c.stride = 0; add("stride-zero", c, 1);
    c = legal; c.stride = 8193; add("stride-above-max", c, 1);
    c = legal; c.stride = c.position; add("end-past-stride", c, 1);
    c = legal; c.position = std::numeric_limits<int>::max();
    c.sequences = std::numeric_limits<int>::max(); add("sum-signed-overflow", c, 1);
    c = legal; c.sequences = std::numeric_limits<int>::max(); add("length-int-max", c, 1);
    c = legal; c.position = std::numeric_limits<int>::min(); add("position-int-min", c, 1);
    c = legal; c.stride = std::numeric_limits<int>::min(); add("stride-int-min", c, 1);
    c = legal; c.valid = legal.valid - 1; add("history-mismatch", c, 2);
    c = legal; c.valid = -1; add("history-negative", c, 2);
    c = legal; c.mask = 2; add("mask-unsupported", c, 3);
    c = legal; c.sequences = 2; c.mask = 0; add("prefill-mask-zero", c, 3);
    c = legal; c.flags = 1; add("flags-one", c, 4);
    c = legal; c.flags = -1; add("flags-negative", c, 4);
    c = legal; c.batches = 0; c.valid = -1; c.mask = 2; c.flags = 1;
    add("precedence-shape", c, 1);
    c = legal; c.valid = -1; c.mask = 2; c.flags = 1;
    add("precedence-history", c, 2);
    c = legal; c.mask = 2; c.flags = 1; add("precedence-mask", c, 3);
    c = legal; c.position = 0; c.valid = 0; c.flags = 1;
    add("invalid-start-must-not-retire", c, 4);
    for (const auto& test : cases) {
        CallBuffers call(2, 2);
        const auto query_before = call.query;
        const auto active_before = call.active;
        const auto output_before = call.output;
        int status = -777;
        invoke(call, live.backing, test.controls, status);
        require(status == test.status, std::string(test.name) + ": wrong first status");
        require(call.query == query_before && call.active == active_before,
                std::string(test.name) + ": input changed");
        require(call.output == output_before, std::string(test.name) + ": output changed");
        compare_backing(live.backing, live.expected_backing, test.name);
        ++rejected_calls;
        std::cout << "REJECTION_PASS " << test.name << '\n';
    }
    // Backing still has stale poison for resident history: this continuation
    // distinguishes unchanged residency from a validation path that retired it.
    live.append(1, 0);
}

void paper_suite() {
    Session paper("paper-and-B2-window", 2, 47, 4, Pattern::Asymmetric, 11);
    paper.append(8);
    for (int p = 8; p <= 21; ++p) paper.append(1, p % 2);
}

void residency_suite() {
    Session batch_one("B1-window-and-host-tiles", 1, 47, 5, Pattern::Uniform, 0);
    batch_one.append(40);
    batch_one.append(1);
    batch_one.append(1, 0);

    Session tiled("B2-host-stride-tail", 2, 23, 6, Pattern::Asymmetric, 19);
    tiled.append(17);
    tiled.append(6);  // Exactly the legal, non-window-multiple stride endpoint.

    // These layer regions have never previously been assigned in this process.
    // No host mutation of earlier logical records occurs between the calls.
    Session cold_two("cold-B2-nonalloc", 2, 47, 7, Pattern::Uniform, 0, 21);
    cold_two.append(1);
    cold_two.append(1, 0);
    Session cold_one("cold-B1-nonalloc", 1, 53, 8, Pattern::Asymmetric, 23, 41);
    cold_one.append(1, 0);
    cold_one.append(1);
}

void isolation_suite() {
    Session layer_a("isolation-layer-A", 1, 47, 10, Pattern::Uniform, 0);
    Session layer_b("isolation-layer-B", 2, 53, 11, Pattern::Asymmetric, 29);
    layer_a.append(3);
    layer_b.append(2);
    layer_a.append(1);
    layer_b.append(1, 0);
    Session new_epoch("retire-new-epoch-and-geometry", 2, 59, 42,
                      Pattern::Asymmetric, 31);
    new_epoch.append(2);
    layer_b.append(1);
    Session same_identity("retire-same-identity", 1, 43, 42, Pattern::Uniform, 0);
    same_identity.append(2);
    same_identity.append(1, 0);
    layer_b.append(1);
}

void identity_suite() {
    Session negative_min("identity-int-min", 1, 8192,
                         std::numeric_limits<int>::min(), Pattern::Asymmetric, 37);
    negative_min.append(2);
    negative_min.append(1, 0);
    Session negative_all("identity-all-bits", 2, 37, -1, Pattern::Uniform, 0);
    negative_all.append(2);
    negative_all.append(1);
}

void full_suite() {
    numeric_suite();
    paper_suite();
    residency_suite();
    isolation_suite();
    identity_suite();
    invalid_suite();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string suite = "cosim";
        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];
            if (argument == "--suite" && i + 1 < argc) suite = argv[++i];
            else throw std::runtime_error("usage: brisk_hls_tb [--suite "
                "cosim|full|numeric|paper|residency|isolation|identity|validation]");
        }
        oracle_self_check();
        if (suite == "full") full_suite();
        else if (suite == "cosim" || suite == "numeric") numeric_suite();
        else if (suite == "paper") paper_suite();
        else if (suite == "residency") residency_suite();
        else if (suite == "isolation") isolation_suite();
        else if (suite == "identity") identity_suite();
        else if (suite == "validation") invalid_suite();
        else throw std::runtime_error("unknown suite: " + suite);
        require(successful_calls >= 2 && successful_calls + rejected_calls >= 3
                && compared_elements != 0,
                "required multi-transaction independent checks were not exercised");
        std::cout << "\n| Suite | Successful calls | Rejected calls | Compared elements |\n"
                  << "| --- | ---: | ---: | ---: |\n| " << suite << " | "
                  << successful_calls << " | " << rejected_calls << " | "
                  << compared_elements << " |\nSIMULATION PASSED\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "SIMULATION FAILED: " << error.what() << '\n';
        return 1;
    }
}
