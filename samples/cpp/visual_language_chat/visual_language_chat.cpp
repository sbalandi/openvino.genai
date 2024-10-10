// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "load_image.hpp"
#include <filesystem>
#include <openvino/genai/vlm_pipeline.hpp>
#include <openvino/runtime/intel_gpu/properties.hpp>
// #include <openvino/genai/generation_config.hpp>

namespace fs = std::filesystem;

bool print_subword(std::string&& subword) {
    return !(std::cout << subword << std::flush);
}

int main(int argc, char* argv[]) try {
    if (3 != argc) {
        throw std::runtime_error(std::string{"Usage "} + argv[0] + " <MODEL_DIR> <IMAGE_FILE>");
    }

    // multinomial or beam_search can be used as well
    ov::genai::GenerationConfig generation_config = ov::genai::greedy();
    // ov::genai::GenerationConfig generation_config = ov::genai::multinomial();
    // ov::genai::GenerationConfig generation_config = ov::genai::beam_search();

    ov::AnyMap properies;
    properies.insert(ov::genai::generation_config(generation_config));

    // streamer could be used with greedy and multinomial
     // if num_return_sequences > 1 in case of multinomial, the streamer will use the output from the first sequence
    if (generation_config.is_greedy_decoding() or generation_config.is_multinomial()) {
        properies.insert(ov::genai::streamer(print_subword));
    }

    std::vector<ov::Tensor> images;
    std::string input_path = argv[2];
    if (!input_path.empty() && fs::exists(input_path)) {
        if (fs::is_directory(input_path)) {
            for (const auto& dir_entry : fs::directory_iterator(input_path)) {
                ov::Tensor image = utils::load_image(dir_entry.path());
                images.push_back(std::move(image));
            }
        } else if (fs::is_regular_file(input_path)) {
            ov::Tensor image = utils::load_image(input_path);
            images.push_back(std::move(image));
        }
    }

    if (images.empty())
        throw std::runtime_error("No one image found by path " + input_path);
    else
        properies.insert(images.size() == 1 ? ov::genai::image(images.at(0)) : ov::genai::images(images));

    std::string device = "CPU";  // GPU can be used as well
    ov::AnyMap enable_compile_cache;
    if ("GPU" == device) {
        // Cache compiled models on disk for GPU to save time on the
        // next run. It's not beneficial for CPU.
        enable_compile_cache.insert({ov::cache_dir("vlm_cache")});
    }
    ov::genai::VLMPipeline pipe(argv[1], device, enable_compile_cache);
    std::string prompt;

    pipe.start_chat();
    std::cout << "question:\n";
    if (!std::getline(std::cin, prompt)) {
        throw std::runtime_error("std::cin failed");
    }
    auto resuls = pipe.generate(prompt, properies);
    if (generation_config.is_beam_search()) {
        std::cout << resuls.texts.at(0) << std::endl;
    }
    std::cout << "\n----------\n"
        "question:\n";
    while (std::getline(std::cin, prompt)) {
        resuls = pipe.generate(prompt, properies);
        if (generation_config.is_beam_search()) {
            std::cout << resuls.texts.at(0) << std::endl;
        }
        std::cout << "\n----------\n"
            "question:\n";
    }
    pipe.finish_chat();
} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
}
