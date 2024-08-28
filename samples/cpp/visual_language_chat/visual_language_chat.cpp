// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "read_image.hpp"
#include <openvino/genai/vlm_pipeline.hpp>
#include <openvino/runtime/intel_gpu/properties.hpp>

namespace {
bool callback(std::string&& subword) {
    return !(std::cout << subword);
}
}

int main(int argc, char* argv[]) try {
    if (3 != argc) {
        throw std::runtime_error(std::string{"Usage "} + argv[0] + " <MODEL_DIR> <IMAGE_FILE>");
    }
    ov::Tensor image = utils::load_image(argv[2]);
    std::string device = "CPU";  // GPU can be used as well
    ov::genai::VLMPipeline pipe(argv[1], device);

    std::string prompt;
    std::cout << "question:\n";
    if (!std::getline(std::cin, prompt)) {
        throw std::runtime_error("std::cin failed");
    }
    pipe.generate({prompt, image}, callback);
    std::cout << "\nquestion:\n";
    while (std::getline(std::cin, prompt)) {
        pipe.generate({prompt}, callback);
        std::cout << "\n----------\n"
            "question:\n";
    }
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
