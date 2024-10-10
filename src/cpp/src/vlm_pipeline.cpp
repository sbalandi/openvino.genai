// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/vlm_pipeline.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "sampler.hpp"
#include "clip.hpp"
#include <openvino/openvino.hpp>
#include "../src/text_callback_streamer.hpp"
#include "utils.hpp"
#include <optional>
#include <random>

using namespace ov::genai;

namespace {
template<class... Ts> struct overloaded : Ts... {using Ts::operator()...;};
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

constexpr size_t BATCH_SIZE = 1;

struct Args {
    bool do_sample = false;
    int top_k = 0;
    float top_p = 0.7f;
    float temp = 0.95f;
    float repeat_penalty = 1.0f;
};

ov::Tensor process_prompt(ov::InferRequest& embedding, const ov::Tensor& prompt, float scale_emb) {
    embedding.set_input_tensor(prompt);
    embedding.infer();

    const ov::Tensor& embed_output_tensor = embedding.get_output_tensor();

    ov::Shape out_shape = embed_output_tensor.get_shape();
    float* data = embed_output_tensor.data<float>();

    //embedding * scale_emb
    for (size_t idx = 0; idx < embed_output_tensor.get_size(); idx++) {
        data[idx] = data[idx] * scale_emb;
    }
    return embed_output_tensor;
}

ov::Tensor concatenate(const ov::Tensor& first, const ov::Tensor& second) {
    size_t res_d_0 = first.get_shape().at(0);
    size_t res_d_1 = first.get_shape().at(1);
    OPENVINO_ASSERT(second.get_shape().at(0) == res_d_0);
    OPENVINO_ASSERT(second.get_shape().at(1) == res_d_1);
    size_t res_d_2 = first.get_shape().at(2) + second.get_shape().at(2);
    ov::Tensor res{first.get_element_type(), {res_d_0, res_d_1, res_d_2}};
    float* first_data = first.data<float>();
    float* second_data = second.data<float>();
    float* res_data = res.data<float>();
    for (size_t i = 0; i < res_d_0; ++i) {
        for (size_t j = 0; j < res_d_1; ++j) {
            size_t k = 0;
            for (; k < first.get_shape().at(2); ++k) {
                res_data[i * res_d_1 * res_d_2 + j * res_d_2 + k]
                    = first_data[i * res_d_1 * first.get_shape().at(2) + j * first.get_shape().at(2) + k];
            }
            for (size_t l = 0; l < second.get_shape().at(2); ++l, ++k) {
                res_data[i * res_d_1 * res_d_2 + j * res_d_2 + k]
                    = second_data[i * res_d_1 * second.get_shape().at(2) + j * second.get_shape().at(2) + l];
            }
        }
    }
    return res;
}

ov::Tensor concatenate_mid_dim(const ov::Tensor& first, const ov::Tensor& second) {
    size_t res_d_0 = first.get_shape().at(0);
    size_t res_d_2 = first.get_shape().at(2);
    OPENVINO_ASSERT(second.get_shape().at(0) == res_d_0);
    OPENVINO_ASSERT(second.get_shape().at(2) == res_d_2);
    size_t res_d_1 = first.get_shape().at(1) + second.get_shape().at(1);
    ov::Tensor res{first.get_element_type(), {res_d_0, res_d_1, res_d_2}};
    float* first_data = first.data<float>();
    float* second_data = second.data<float>();
    float* res_data = res.data<float>();
    for (size_t i = 0; i < res_d_0; ++i) {
        size_t j = 0;
        for (; j < first.get_shape().at(1); ++j) {
            std::copy_n(
                first_data + i * first.get_shape().at(1) * res_d_2 + j * res_d_2,
                res_d_2,
                res_data + i * res_d_1 * res_d_2 + j * res_d_2
            );
        }
        for (size_t k = 0; k < second.get_shape().at(1); ++k, ++j) {
            std::copy_n(
                second_data + i * second.get_shape().at(1) * res_d_2 + k * res_d_2,
                res_d_2,
                res_data + i * res_d_1 * res_d_2 + j * res_d_2
            );
        }
    }
    return res;
}

/// embed_dim: output dimension for each position
/// pos: a list of positions to be encoded: size (H, W)
/// out: (H, W, D)
ov::Tensor get_1d_sincos_pos_embed_from_grid_new(size_t embed_dim, const ov::Tensor& pos) {
    OPENVINO_ASSERT(embed_dim % 2 == 0);
    OPENVINO_ASSERT(pos.get_shape().size() == 3);
    OPENVINO_ASSERT(pos.get_shape().at(0) == 1);
    size_t d0 = pos.get_shape().at(1);
    size_t d1 = pos.get_shape().at(2);
    size_t d2 = embed_dim / 2;
    std::vector<float> omega(d2);
    for (size_t idx = 0; idx < omega.size(); ++idx) {
        omega.at(idx) = idx / (embed_dim / 2.0f);
        omega.at(idx) = 1.0f / std::pow(10000.0f, omega.at(idx));  // (D/2,)
    }
    const float* const pos_data = pos.data<float>();
    ov::Tensor out(ov::element::f32, {d0, d1, d2});  // (H, W, D/2), outer product
    float* out_data = out.data<float>();
    for (size_t i = 0; i < d0; ++i) {
        for (size_t j = 0; j < d1; ++j) {
            for (size_t k = 0; k < d2; ++k) {
                out_data[i * d1 * d2 + j * d2 + k]
                    = pos_data[i * d1 + j] * omega[k];
            }
        }
    }

    ov::Tensor emb_sin{out.get_element_type(), out.get_shape()};  // (H, W, D/2)
    float* emb_sin_data = emb_sin.data<float>();
    std::transform(out_data, out_data + out.get_size(), emb_sin_data, [](float arg) {
        return std::sin(arg);
    });
    ov::Tensor emb_cos{out.get_element_type(), out.get_shape()};  // (H, W, D/2)
    float* emb_cos_data = emb_cos.data<float>();
    std::transform(out_data, out_data + out.get_size(), emb_cos_data, [](float arg) {
        return std::cos(arg);
    });
    return concatenate(emb_sin, emb_cos); // (H, W, D)
}

ov::Tensor get_2d_sincos_pos_embed_from_grid(size_t embed_dim, const ov::Tensor& grid) {
    OPENVINO_ASSERT(embed_dim % 2 == 0);
    // use half of dimensions to encode grid_h
    ov::Coordinate begin_h{0, 0, 0};
    ov::Coordinate end_h{grid.get_shape()};
    end_h.at(0) = 1;
    ov::Coordinate begin_w{1, 0, 0};
    ov::Coordinate end_w{grid.get_shape()};
    end_w.at(0) = 2;
    ov::Tensor emb_h = get_1d_sincos_pos_embed_from_grid_new(embed_dim / 2, ov::Tensor{grid, begin_h, end_h});  // (H, W, D/2)
    ov::Tensor emb_w = get_1d_sincos_pos_embed_from_grid_new(embed_dim / 2, ov::Tensor{grid, begin_w, end_w});  // (H, W, D/2)
    return concatenate(emb_h, emb_w);
}

/// image_size: image_size or (image_height, image_width)
/// return:
/// pos_embed: [image_height, image_width, embed_dim]
ov::Tensor get_2d_sincos_pos_embed(size_t embed_dim, const HeightWidth& image_size) {
    size_t grid_h_size = image_size.height, grid_w_size = image_size.width;
    ov::Tensor grid(ov::element::f32, {2, grid_h_size, grid_w_size});
    float* data = grid.data<float>();
    for (size_t y = 0; y < grid_h_size; ++y) {
        std::iota(data, data + grid_w_size, 0.0f);
        data += grid_w_size;
    }
    for (float y = 0.0f; y < grid_h_size; ++y) {
        std::fill(data, data + grid_w_size, y);
        data += grid_w_size;
    }
    return get_2d_sincos_pos_embed_from_grid(embed_dim, grid);
}

void adjust_pos_cache(
    const std::vector<HeightWidth>& target_sizes,
    size_t hidden_size,
    ov::Tensor& pos_embed_cache
) {
    size_t max_h = std::max_element(target_sizes.begin(), target_sizes.end(), [](const HeightWidth& left, const HeightWidth& right) {
        return left.height < right.height;
    })->height;
    size_t max_w = std::max_element(target_sizes.begin(), target_sizes.end(), [](const HeightWidth& left, const HeightWidth& right) {
        return left.width < right.width;
    })->width;
    size_t allocated_height, allocated_width;
    if (pos_embed_cache) {
        const ov::Shape& allocated_shape = pos_embed_cache.get_shape();
        allocated_height = allocated_shape.at(0);
        allocated_width = allocated_shape.at(1);
    } else {
        allocated_height = allocated_width = 70;
    }
    if (max_h > allocated_height || max_w > allocated_width) {
        allocated_height = std::max(max_h, allocated_height);
        allocated_width = std::max(max_w, allocated_width);
        pos_embed_cache = get_2d_sincos_pos_embed(
            hidden_size, {allocated_height, allocated_width}
        );
    }
}

ov::Tensor resample(VLMPipeline& pipe, const ov::Tensor& encoded_image, const std::vector<HeightWidth>& target_sizes) {
    size_t bs = encoded_image.get_shape().at(0);
    std::vector<size_t> patch_len{target_sizes.size()};
    std::transform(target_sizes.begin(), target_sizes.end(), patch_len.begin(), [](const HeightWidth& height_width) {
        return height_width.height * height_width.width;
    });
    adjust_pos_cache(
        target_sizes,
        pipe.m_vlm_config.hidden_size,
        pipe.m_pos_embed_cache
    );
    size_t max_patch_len = *std::max_element(patch_len.begin(), patch_len.end());
    ov::Tensor key_padding_mask(ov::element::boolean, {bs, max_patch_len});
    bool* mask_data = key_padding_mask.data<bool>();
    size_t embed_len = pipe.m_pos_embed_cache.get_shape().at(2);
    ov::Tensor pos_embed(ov::element::f32, {max_patch_len, bs, embed_len});  // BLD => L * B * D
    float* pos_embed_data = pos_embed.data<float>();
    float* cache_data = pipe.m_pos_embed_cache.data<float>();
    size_t _d0 = pipe.m_pos_embed_cache.get_shape().at(0);
    size_t _d1 = pipe.m_pos_embed_cache.get_shape().at(1);
    for (size_t i = 0; i < bs; ++i) {
        size_t target_h = target_sizes.at(i).height;
        size_t target_w = target_sizes.at(i).width;
        for (size_t h_idx = 0; h_idx < target_h; ++h_idx) {
            for (size_t w_idx = 0; w_idx < target_w; ++w_idx) {
                std::copy_n(
                    cache_data + h_idx * _d1 + w_idx,
                    embed_len,
                    pos_embed_data + (h_idx * target_w + w_idx) * bs * embed_len + i * embed_len
                );
            }
        }
        for (size_t flat = target_h * target_w; flat < max_patch_len; ++flat) {
            std::fill_n(pos_embed_data + flat * bs * embed_len + i * embed_len, embed_len, 0.0f);
        }
        std::fill_n(mask_data + i * max_patch_len, patch_len[i], false);
        std::fill_n(mask_data + i * max_patch_len + patch_len[i], max_patch_len - patch_len[i], true);
    }
    pipe.m_resampler.set_tensor("x", encoded_image);  // [N, H*W, old_hidden_size]
    pipe.m_resampler.set_tensor("pos_embed", pos_embed);  // [H*W, N, new_hidden_size]
    pipe.m_resampler.set_tensor("key_padding_mask", key_padding_mask);  // [N, H*W]
    pipe.m_resampler.infer();
    return pipe.m_resampler.get_output_tensor();  // [N, query_num, new_hidden_size]
}

EncodedGenerationResult get_lm_encoded_results(
    ov::InferRequest& language,
    ov::InferRequest& embedding,
    ov::Tensor inputs_embeds,
    const VLMConfig m_vlm_config,
    const std::shared_ptr<StreamerBase> streamer_ptr,
    Sampler& sampler,
    std::vector<SequenceGroup::Ptr> requests
) {
    SequenceGroup::Ptr request = requests.back();
    GenerationHandle generation = std::make_shared<GenerationHandleImpl>(request->get_generation_stream(), request->get_sampling_parameters());

    language.set_tensor("inputs_embeds", inputs_embeds);

    size_t history_len = language.get_tensor("attention_mask").get_shape().at(1);
    language.get_tensor("attention_mask").set_shape({1, history_len + inputs_embeds.get_shape()[1]});
    std::fill_n(language.get_tensor("attention_mask").data<int64_t>(), language.get_tensor("attention_mask").get_size(), 1);

    language.get_tensor("position_ids").set_shape({1, inputs_embeds.get_shape().at(1)});
    std::iota(language.get_tensor("position_ids").data<int64_t>(), language.get_tensor("position_ids").data<int64_t>() + language.get_tensor("position_ids").get_size(), history_len);

    language.get_tensor("beam_idx").set_shape({ BATCH_SIZE });
    language.get_tensor("beam_idx").data<int32_t>()[0] = 0;

    language.infer();

    int64_t sequence_len = language.get_tensor("logits").get_shape().at(1);
    request->schedule_tokens(sequence_len);

    SamplerOutput sampler_output = sampler.sample(requests, language.get_tensor("logits"));

    language.get_tensor("inputs_embeds").set_shape({BATCH_SIZE, 1, m_vlm_config.hidden_size});
    language.get_tensor("position_ids").set_shape({ BATCH_SIZE, 1 });

    while (!request->has_finished()) {
        request->schedule_tokens(1);
        size_t num_sequences = request->num_running_seqs();
        size_t total_num_tokens = request->get_num_scheduled_tokens() * num_sequences;

        ov::Tensor
            input_ids(ov::element::i64, {total_num_tokens, 1}),
            position_ids(ov::element::i64, {total_num_tokens, 1}),
            beam_idx(ov::element::i32, { total_num_tokens });

        int64_t
            * input_ids_data = input_ids.data<int64_t>(),
            * position_ids_data = position_ids.data<int64_t>();

        size_t num_scheduled_tokens = request->get_num_scheduled_tokens();
        size_t group_position_id = request->get_num_processed_tokens();
        for (Sequence::Ptr& sequence : request->get_running_sequences()) {
            for (size_t token_id = 0, position_id = group_position_id; token_id < num_scheduled_tokens; ++token_id, ++position_id) {
                // compute token for current sequence
                input_ids_data[token_id] = position_id < request->get_prompt_len() ?
                    request->get_prompt_ids()[position_id] :
                    sequence->get_generated_ids()[position_id - request->get_prompt_len()];

                position_ids_data[token_id] = position_id;
            }
            // apply strides to shift to a next sequence
            input_ids_data += num_scheduled_tokens;
            position_ids_data += num_scheduled_tokens;
        }

        embedding.set_input_tensor(input_ids);

        embedding.infer();
        const ov::Tensor& embed_prompt_tensor = embedding.get_output_tensor();
        float* embed_data = embed_prompt_tensor.data<float>();
        for (auto idx = 0; idx < embed_prompt_tensor.get_size(); idx++) {
            embed_data[idx] = embed_data[idx] * m_vlm_config.scale_emb;
        }

        language.set_tensor("inputs_embeds", embed_prompt_tensor);

        language.get_tensor("attention_mask").set_shape({ total_num_tokens, language.get_tensor("attention_mask").get_shape()[1] + 1 });
        std::fill_n(language.get_tensor("attention_mask").data<int64_t>(), language.get_tensor("attention_mask").get_size(), 1);

        language.set_tensor("position_ids", position_ids);

        std::vector<int32_t> beam_idxs = sampler.get_beam_idxs(request->get_request_id());
        int32_t *beam_idx_data = beam_idx.data<int32_t>();
        if (total_num_tokens > beam_idxs.size()) {
            std::fill_n(beam_idx_data, total_num_tokens, 0);
        } else {
            copy(beam_idxs.begin(), beam_idxs.end(), beam_idx_data);
        }
        language.set_tensor("beam_idx", beam_idx);

        language.infer();

        if (streamer_ptr) {
            // first sequence
            int64_t out_token = request.get()->operator[](0)->get_generated_ids().back();
            if (streamer_ptr->put(out_token)) {
                break;
            }
        }

        sampler_output = sampler.sample(requests, language.get_tensor("logits"));
    }

    if (streamer_ptr) {
        streamer_ptr->end();
    }

    EncodedGenerationResult result;
    result.m_request_id = 1;
    std::vector<GenerationOutput> generation_outputs = generation->read_all();
    std::sort(generation_outputs.begin(), generation_outputs.end(), [=] (GenerationOutput& r1, GenerationOutput& r2) {
        return r1.score > r2.score;
    });

    auto num_outputs = std::min(request->get_sampling_parameters().num_return_sequences, generation_outputs.size());
    for (size_t generation_output_idx = 0; generation_output_idx < num_outputs; ++generation_output_idx) {
        const auto& generation_output = generation_outputs[generation_output_idx];
        result.m_generation_ids.push_back(std::move(generation_output.generated_ids));
        result.m_scores.push_back(generation_output.score);
    }
    result.m_status = generation->get_status();

    return result;
}
} // anonymous


class ov::genai::VLMPipeline::VLMPipelineImpl {
};

VLMPipeline::VLMPipeline(
    const std::filesystem::path& model_dir,
    const Tokenizer& tokenizer,
    const std::string& device,
    const ov::AnyMap device_config,
    ov::Core core
) :
    m_vlm_config{
        utils::from_config_json_if_exists<ov::genai::VLMConfig>(
            model_dir, "config.json"
        )
    },
    m_tokenizer{tokenizer},
    m_vision_encoder(model_dir, device, device_config, core),
    m_resampler{core.compile_model(
        model_dir / "resampler.xml", device, device_config
    ).create_infer_request()},
    m_embedding{core.compile_model(
        model_dir / "embed_tokens.xml", device, device_config
    ).create_infer_request()},
    m_language{core.compile_model(
        model_dir / "language_model.xml", device, device_config
    ).create_infer_request()},
    m_pos_embed_cache{
        get_2d_sincos_pos_embed(m_vlm_config.hidden_size, {70, 70})
    },
    m_is_chat_conversation{false} {
        m_language.get_tensor("attention_mask").set_shape({1, 0});
    }

ov::genai::VLMPipeline::~VLMPipeline() = default;

DecodedResults VLMPipeline::generate(
    const std::string& prompt,
    const std::vector<ov::Tensor>& rgbs,
    const GenerationConfig& generation_config,
    const StreamerVariant& streamer
) {
    std::string images_prompt;
    std::vector<EncodedImage> embeds;
    for (const ov::Tensor& rgb : rgbs) {
        EncodedImage encoded_image = m_vision_encoder.encode(rgb);
        if (m_vlm_config.use_image_id) {
            images_prompt += m_vlm_config.im_id_start + std::to_string(image_id) + m_vlm_config.im_id_end;
            ++image_id;
        }
        std::string unk64;
        for (size_t idx = 0; idx < m_vlm_config.query_num; ++idx) {
            unk64 += m_vlm_config.unk;
        }
        images_prompt += m_vlm_config.im_start + unk64 + m_vlm_config.im_end;
        if (encoded_image.slices) {
            ov::Shape slices_shape = encoded_image.slices.get_shape();
            for (size_t row_idx = 0; row_idx < slices_shape.at(0); ++row_idx) {
                for (size_t col_idx = 0; col_idx < slices_shape.at(1); ++col_idx) {
                    images_prompt += m_vlm_config.slice_start + unk64 + m_vlm_config.slice_end;
                }
                images_prompt += '\n';
            }
        }
        if ('\n' != *(images_prompt.end() - 1)) {
            // Image wasn't sliced, add \n to the end of image anyway.
            // Strangely, \n isn't placed between </image><slice>.
            images_prompt += '\n';
        }
        embeds.push_back(std::move(encoded_image));
    }
    images_prompt += prompt;
    ov::Tensor encoded_input;
    ov::Tensor new_chat_tokens;
    size_t prev_tokens_num = 0;
    if (m_is_chat_conversation) {
        // KV cache in model already contains prompts and answers from previous iterations.
        // So only new prompt wrapped into chat template to be sent into model. Tokenizer always returns
        // token_ids = {<bos token>, ...<valuable tokens>}. So if tokenizer applies only to the new prompt,
        // <bos token> will be inserted on every iteration.
        // So actual pipeline calculates input_ids for whole chat history + for whole chat history without the new prompt
        // and takes only the difference between them.
        // The chat history cannot be saved as already encoded tokens because generate call doesn't return <eos> token, but
        // KV cache contains it. So we have to add it manually or get it by tokenization all chat history.
        m_history.push_back({{"role", "user"}, {"content", images_prompt}});
        constexpr bool add_generation_prompt = true;
        std::string new_templated_chat_history = m_tokenizer.apply_chat_template(m_history, add_generation_prompt);
        new_chat_tokens = m_tokenizer.encode(new_templated_chat_history).input_ids;
        if (0 == m_language.get_tensor("attention_mask").get_shape().at(1)) {
            encoded_input = new_chat_tokens;
        } else {
            TokenizedInputs prev_chat_tokens = m_tokenizer.encode(
                m_templated_chat_history
            );
            encoded_input = utils::subtract_chat_tokenized_inputs(
                {new_chat_tokens}, prev_chat_tokens
            ).input_ids;
            prev_tokens_num = prev_chat_tokens.input_ids.get_size();
        }
        m_templated_chat_history = std::move(new_templated_chat_history);
    } else {
        encoded_input = m_tokenizer.encode(images_prompt).input_ids;
    }
    m_embedding.set_input_tensor(encoded_input);
    m_embedding.infer();
    ov::Tensor inputs_embeds = m_embedding.get_output_tensor();
    OPENVINO_ASSERT(
        m_vlm_config.hidden_size == inputs_embeds.get_shape().at(2),
        "Unexpected embedding size"
    );
    ov::Tensor special_tokens = m_tokenizer.encode(
        m_vlm_config.im_start
        + m_vlm_config.im_end
        + m_vlm_config.slice_start
        + m_vlm_config.slice_end
    ).input_ids;
    OPENVINO_ASSERT(
        4 == special_tokens.get_shape().at(1),
        "Every special token must be represented with a single int."
    );
    int64_t im_start_id = special_tokens.data<int64_t>()[0];
    int64_t im_end_id = special_tokens.data<int64_t>()[1];
    int64_t slice_start_id = special_tokens.data<int64_t>()[2];
    int64_t slice_end_id = special_tokens.data<int64_t>()[3];
    int64_t im_start_pos = 0, slice_start_pos = 0;
    int64_t* begin = encoded_input.data<int64_t>();
    int64_t* ids = begin;
    size_t encoded_input_size = encoded_input.get_size();
    int64_t* end = ids + encoded_input_size;
    float* inputs_embeds_data = inputs_embeds.data<float>();
    for (const EncodedImage& encoded_image : embeds) {
        const ov::Tensor& resampled_source = resample(*this, encoded_image.resized_source, {encoded_image.resized_source_size});
        float* emb = resampled_source.data<float>();
        ids = std::find(ids, end, im_start_id);
        OPENVINO_ASSERT(end != ids);
        std::copy_n(emb, resampled_source.get_size(), inputs_embeds_data + std::distance(begin, ids) * m_vlm_config.hidden_size);
        ids += m_vlm_config.query_num;
        if (encoded_image.slices) {
            size_t token_idx = 0;
            const ov::Shape& slices_shape = encoded_image.slices.get_shape();
            const std::vector<HeightWidth>& sliced_sizes = encoded_image.slices_sizes;
            for (size_t i = 0; i < slices_shape.at(0); ++i) {
                for (size_t ja = 0; ja < slices_shape.at(1); ++ja) {
                    size_t d2 = slices_shape.at(2);
                    size_t d3 = slices_shape.at(3);
                    ov::Tensor encoded_view{ov::element::f32, {1, d2, d3}, encoded_image.slices.data<float>() + (i * slices_shape.at(1) + ja) * d2 * d3};
                    const ov::Tensor& vision_embed_tensor_i_j = resample(*this, encoded_view, {sliced_sizes.at(i * slices_shape.at(1) + ja)});
                    ids = std::find(ids, end, slice_start_id);
                    OPENVINO_ASSERT(end != ids);
                    std::copy_n(vision_embed_tensor_i_j.data<float>(), vision_embed_tensor_i_j.get_size(), inputs_embeds_data + std::distance(begin, ids) * m_vlm_config.hidden_size);
                    ids += m_vlm_config.query_num;
                }
            }
        }
    }

    Sampler sampler = Sampler(m_tokenizer);

    std::vector<SequenceGroup::Ptr> requests;
    // request_id, input_ids, generation_config, block_size, enable_prefix_caching
    // now we have one prompt as input, so we need one request
    SequenceGroup::Ptr sequence_group = std::make_shared<SequenceGroup>(0, prev_tokens_num == 0 ? encoded_input : new_chat_tokens, generation_config, 1, false);
    sequence_group->update_processed_tokens_num(prev_tokens_num);
    sequence_group->set_sequence_group_ptr(sequence_group);
    requests.push_back(sequence_group);

    std::shared_ptr<StreamerBase> streamer_ptr = std::visit(overloaded{
        [&m_tokenizer = m_tokenizer](
            const std::function<bool(std::string)>& callback
        ) -> std::shared_ptr<StreamerBase> {
            return std::make_shared<TextCallbackStreamer>(m_tokenizer, callback);
        },
        [](const std::shared_ptr<StreamerBase>& ptr) {
            return ptr;
        },
        [](std::monostate) {
            return std::shared_ptr<StreamerBase>{nullptr};
        },
    }, streamer);

    if ((!(generation_config.is_greedy_decoding() || generation_config.is_multinomial())) && streamer_ptr) {
        OPENVINO_THROW("Currently streaming is possible only for greedy or multinomial decoding");
    }

    EncodedGenerationResult encoded_result = get_lm_encoded_results(m_language, m_embedding, inputs_embeds, m_vlm_config, streamer_ptr, sampler, requests);

    DecodedResults decoded;
    for (size_t idx = 0; idx < encoded_result.m_generation_ids.size(); ++idx) {
        decoded.texts.push_back(m_tokenizer.decode(encoded_result.m_generation_ids.at(idx)));
        decoded.scores.push_back(encoded_result.m_scores.at(idx));
    }

    std::string decoded_results = decoded.texts.at(0);
    if (m_is_chat_conversation) {
        // Tail of chat template is missing in KV cache.
        // Find the tail to concatenate it with the next input prompt.
        m_templated_chat_history.append(decoded_results);
        m_history.push_back({{"role", "assistant"}, {"content", decoded_results}});
    } else {
        for (auto& variable : m_language.query_state()) {
            variable.reset();
        }
        m_language.get_tensor("attention_mask").set_shape({1, 0});
    }
    return decoded;
}

DecodedResults VLMPipeline::generate(
    const std::string& prompt,
    const ov::AnyMap& config_map
) {
    auto image = config_map.find(ov::genai::image.name());
    auto images = config_map.find(ov::genai::images.name());
    OPENVINO_ASSERT(
        config_map.end() == image || config_map.end() == images,
        "Only one property can be set: image of images."
    );
    std::vector<ov::Tensor> rgbs;
    if (config_map.end() != image) {
        rgbs = {image->second.as<ov::Tensor>()};
    } if (config_map.end() != images) {
        rgbs = images->second.as<std::vector<ov::Tensor>>();
    }
    ov::genai::OptionalGenerationConfig config_arg = utils::get_config_from_map(config_map);
    GenerationConfig config = (config_arg.has_value()) ? *config_arg : get_generation_config();
    config.update_generation_config(config_map);

    // If eos_token_id was not provided, take value
    if (config.eos_token_id == -1)
        config.set_eos_token_id(m_tokenizer.get_eos_token_id());

    return generate(
        prompt,
        rgbs,
        config,
        utils::get_streamer_from_map(config_map)
    );
}

void VLMPipeline::start_chat(const std::string& system_message) {
    m_is_chat_conversation = true;
    bool have_state = 0 != m_language.get_tensor("attention_mask").get_size();
    if (have_state) {
        // Resetting state may be slow.
        for (ov::VariableState& variable : m_language.query_state()) {
            variable.reset();
        }
        // Since if is already introduced, move all resetting here.
        m_language.get_tensor("attention_mask").set_shape({1, 0});
        m_history.clear();
        m_templated_chat_history.clear();
    }
    if (system_message.empty()) {
        return;
    }
    m_history = {{{"role", "system"}, {"content", system_message}}};
    constexpr bool add_generation_prompt = false;
    m_templated_chat_history = m_tokenizer.apply_chat_template(m_history, add_generation_prompt);
}

void VLMPipeline::set_chat_template(const std::string& new_template) {
    m_tokenizer.set_chat_template(new_template);
}

GenerationConfig VLMPipeline::get_generation_config() const {
    return m_generation_config;
}

void VLMPipeline::set_generation_config(const GenerationConfig& new_config) {
    m_generation_config = new_config;
}
