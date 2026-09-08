#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "image_dataset.hpp"
#include "image_filters.hpp"

#include <filesystem>
#include <iostream>
#include <vector>

namespace fs = std::filesystem;

struct MorphOperation {
    std::string name;
    int type;
};

int main() {
    const fs::path input_folder_path = "../images/";
    const fs::path output_folder_path = "../output_images/morphological_operators/";

    constexpr int kernel_size_3 = 3;
    constexpr int kernel_size_5 = 5;
    constexpr int rank = 4;

    const cv::Mat morph_kernel_3 = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(kernel_size_3, kernel_size_3));
    const std::vector<MorphOperation> morph_operations = {
        {"eroded", cv::MORPH_ERODE},
        {"dilated", cv::MORPH_DILATE},
        {"opening", cv::MORPH_OPEN},
        {"closing", cv::MORPH_CLOSE},
        {"gradient", cv::MORPH_GRADIENT},
        {"top_hat", cv::MORPH_TOPHAT},
        {"black_hat", cv::MORPH_BLACKHAT},
    };

    for (const auto& file_name : noisy_image_filenames()) {
        const cv::Mat image = cv::imread((input_folder_path / file_name).string(), cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "Warning: Could not read image: " << file_name << '\n';
            continue;
        }

        cv::Mat median_3x3, median_5x5, nrank_3x3, nrank_5x5;
        cv::medianBlur(image, median_3x3, kernel_size_3);
        cv::medianBlur(image, median_5x5, kernel_size_5);
        n_rank_filter(image, nrank_3x3, kernel_size_3, rank);
        n_rank_filter(image, nrank_5x5, kernel_size_5, rank);

        const std::vector<std::pair<std::string, cv::Mat>> first_stage_outputs = {
            {"median3x3", median_3x3},
            {"median5x5", median_5x5},
            {"nrank3", nrank_3x3},
            {"nrank5", nrank_5x5},
        };

        for (const auto& [filter_name, filtered_image] : first_stage_outputs) {
            for (const auto& operation : morph_operations) {
                cv::Mat morph_image;
                cv::morphologyEx(filtered_image, morph_image, operation.type, morph_kernel_3);

                const fs::path output_directory = output_folder_path / filter_name / operation.name;
                fs::create_directories(output_directory);
                const fs::path output_path = output_directory / file_name;
                if (!cv::imwrite(output_path.string(), morph_image)) {
                    std::cerr << "Warning: Could not write image: " << output_path << '\n';
                }
            }
        }
    }
    return 0;
}