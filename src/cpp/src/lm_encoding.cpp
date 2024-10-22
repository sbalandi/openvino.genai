// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <regex>
#include <vector>

#include "lm_encoding.hpp"
#include "openvino/genai/perf_metrics.hpp"


namespace ov {
namespace genai {

// void reset_all_inputs_to_empty_tensors(ov::InferRequest& request) {
//     request.set_tensor("input_ids", ov::Tensor(ov::element::i64, {0, 0}));
//     request.set_tensor("beam_idx", ov::Tensor(ov::element::i32, {0}));
//     if (request.get_compiled_model().inputs().size() == 4)
//         request.set_tensor("position_ids", ov::Tensor(ov::element::i64, {0, 0}));
// }


EncodedResults get_lm_encoded_results(
    ov::InferRequest& m_model_runner,
    const ov::Tensor& input_ids,
    ov::Tensor attention_mask,
    const std::shared_ptr<StreamerBase>& streamer_ptr,
    Sampler& sampler,
    std::vector<SequenceGroup::Ptr> sequence_groups,
    std::optional<ov::Tensor> position_ids,
    std::optional<ov::InferRequest> m_embedding,
    std::optional<size_t> hidden_size,
    std::optional<float> scale_emb
) {
    std::vector<GenerationHandle> generations;
    for (SequenceGroup::Ptr sequence_group : sequence_groups) {
        generations.push_back(std::make_shared<GenerationHandleImpl>(sequence_group->get_generation_stream(), sequence_group->get_sampling_parameters()));
    }

    ov::Shape prompts_shape = input_ids.get_shape();
    const size_t batch_size = prompts_shape[0];

    const size_t prompt_len = prompts_shape[1];
    const size_t max_new_tokens = sequence_groups.at(0)->get_sampling_parameters().get_max_new_tokens(prompt_len);

    // Initialize results and performance metrics.
    EncodedResults results;
    auto& raw_perf_counters = results.perf_metrics.raw_metrics;
    raw_perf_counters.m_new_token_times.reserve(max_new_tokens);
    raw_perf_counters.m_batch_sizes.reserve(max_new_tokens);
    raw_perf_counters.m_token_infer_durations.reserve(max_new_tokens);
    raw_perf_counters.m_inference_durations = {{ MicroSeconds(0.0f) }};

    // Initialize inputs
    if (m_embedding.has_value())
        m_model_runner.set_tensor("inputs_embeds", input_ids);
    else
        m_model_runner.set_tensor("input_ids", input_ids);

    m_model_runner.set_tensor("attention_mask", attention_mask);
    
    if (position_ids.has_value())
        m_model_runner.set_tensor("position_ids", *position_ids);

    m_model_runner.get_tensor("beam_idx").set_shape({ batch_size });
    std::fill_n(m_model_runner.get_tensor("beam_idx").data<int32_t>(), batch_size, 0);

    const auto infer_start = std::chrono::steady_clock::now();
    m_model_runner.infer();
    const auto infer_end = std::chrono::steady_clock::now();
    const auto infer_ms = PerfMetrics::get_microsec(infer_end - infer_start);
    raw_perf_counters.m_inference_durations[0] += MicroSeconds(infer_ms);
    raw_perf_counters.m_token_infer_durations.emplace_back(infer_ms);
    raw_perf_counters.m_new_token_times.emplace_back(infer_end);
    raw_perf_counters.m_batch_sizes.emplace_back(batch_size);

    auto logits = m_model_runner.get_tensor("logits");

    int64_t sequence_len = logits.get_shape().at(1);
    for (auto& sequence_group : sequence_groups)
        sequence_group->schedule_tokens(sequence_len);

    SamplerOutput sampler_output = sampler.sample(sequence_groups, logits);
    
    if (m_embedding.has_value() && hidden_size.has_value()) {
        m_model_runner.get_tensor("inputs_embeds").set_shape({ batch_size, 1, *hidden_size });
        m_model_runner.get_tensor("position_ids").set_shape({ batch_size, 1 });
    }

    bool should_continue = std::none_of(sequence_groups.begin(), sequence_groups.end(), [](SequenceGroup::CPtr sg) { return sg->has_finished(); });
    
    while (should_continue) {
        size_t num_sequence_groups = sequence_groups.size();
        size_t batch_size_in_sequences = 0;
        size_t total_num_tokens = 0;
        size_t max_context_len_val = 0;

        for (auto& sequence_group : sequence_groups) {
            sequence_group->schedule_tokens(1);

            // compute aggregated values
            size_t num_sequences = sequence_group->num_running_seqs();
            batch_size_in_sequences += num_sequences;
            total_num_tokens += sequence_group->get_num_scheduled_tokens() * num_sequences;
            max_context_len_val = std::max(max_context_len_val, sequence_group->get_context_len());
        }

        ov::Tensor
            new_input_ids(ov::element::i64, {total_num_tokens, 1}),
            new_position_ids(ov::element::i64, {total_num_tokens, 1}),
            new_beam_idx(ov::element::i32, { total_num_tokens });

        int64_t
            * input_ids_data = new_input_ids.data<int64_t>(),
            * position_ids_data = new_position_ids.data<int64_t>();
        int32_t
            * beam_idx_data = new_beam_idx.data<int32_t>();

        for (auto& sequence_group : sequence_groups) {
            std::vector<Sequence::Ptr> running_sequences = sequence_group->get_running_sequences();
            size_t num_running_sequences = running_sequences.size();
            size_t num_scheduled_tokens = sequence_group->get_num_scheduled_tokens();
            size_t group_position_id = sequence_group->get_num_processed_tokens();

            for (size_t seq_id = 0; seq_id < num_running_sequences; ++seq_id) {
                Sequence::CPtr sequence = running_sequences[seq_id];

                for (size_t token_id = 0, position_id = group_position_id; token_id < num_scheduled_tokens; ++token_id, ++position_id) {
                    // compute token for current sequence
                    input_ids_data[token_id] = position_id < sequence_group->get_prompt_len() ?
                        sequence_group->get_prompt_ids()[position_id] :
                        sequence->get_generated_ids()[position_id - sequence_group->get_prompt_len()];

                    position_ids_data[token_id] = position_id;
                }

                // apply strides to shift to a next sequence
                input_ids_data += num_scheduled_tokens;
                position_ids_data += num_scheduled_tokens;
            }

            std::vector<int32_t> beam_idxs = sampler.get_beam_idxs(sequence_group);
            copy(beam_idxs.begin(), beam_idxs.end(), beam_idx_data);
            beam_idx_data += beam_idxs.size();
        }

        if (m_embedding.has_value() && scale_emb.has_value()) {
            auto embedding = *m_embedding;
            embedding.set_input_tensor(new_input_ids);

            embedding.infer();
            const ov::Tensor& embed_prompt_tensor = embedding.get_output_tensor();
            float* embed_data = embed_prompt_tensor.data<float>();
            for (auto idx = 0; idx < embed_prompt_tensor.get_size(); idx++) {
                embed_data[idx] = embed_data[idx] * *scale_emb;
            }

            m_model_runner.set_tensor("inputs_embeds", embed_prompt_tensor);
        } else {
            m_model_runner.set_tensor("input_ids", new_input_ids);
        }

        m_model_runner.get_tensor("attention_mask").set_shape({ total_num_tokens, m_model_runner.get_tensor("attention_mask").get_shape()[1] + 1 });
        std::fill_n(m_model_runner.get_tensor("attention_mask").data<int64_t>(), m_model_runner.get_tensor("attention_mask").get_size(), 1);

        if (position_ids.has_value())
            m_model_runner.set_tensor("position_ids", new_position_ids);

        m_model_runner.set_tensor("beam_idx", new_beam_idx);

        const auto infer_start = std::chrono::steady_clock::now();
        m_model_runner.infer();
        const auto infer_end = std::chrono::steady_clock::now();
        const auto infer_ms = PerfMetrics::get_microsec(infer_end - infer_start);
        raw_perf_counters.m_inference_durations[0] += MicroSeconds(infer_ms);
        raw_perf_counters.m_token_infer_durations.emplace_back(infer_ms);
        raw_perf_counters.m_new_token_times.emplace_back(infer_end);
        raw_perf_counters.m_batch_sizes.emplace_back(batch_size);


        if (streamer_ptr) {
            // stream data from first sequence
            int64_t out_token = sequence_groups.at(0).get()->operator[](0)->get_generated_ids().back();
            if (streamer_ptr->put(out_token)) {
                break;
            }
        }

        sampler_output = sampler.sample(sequence_groups, m_model_runner.get_tensor("logits"));

        should_continue = std::none_of(sequence_groups.begin(), sequence_groups.end(), [](SequenceGroup::CPtr sg) { return sg->has_finished(); });
    }

    if (streamer_ptr) {
        streamer_ptr->end();
    }

    // reset_all_inputs_to_empty_tensors(m_model_runner);

    for (size_t i = 0; i < sequence_groups.size(); i++) {
        auto request = sequence_groups[i];
        auto generation_outputs = generations[i]->read_all();

        std::sort(generation_outputs.begin(), generation_outputs.end(), [] (const GenerationOutput& r1, const GenerationOutput& r2) {
            return r1.score > r2.score;
        });

        auto num_outputs = std::min(request->get_sampling_parameters().num_return_sequences, generation_outputs.size());
        for (size_t generation_output_idx = 0; generation_output_idx < num_outputs; ++generation_output_idx) {
            const auto& generation_output = generation_outputs[generation_output_idx];
            results.tokens.push_back(std::move(generation_output.generated_ids));
            results.scores.push_back(generation_output.score);
        }
    }

    return results;
}

}  // namespace genai
}  // namespace ov
