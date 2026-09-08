#include <opencv2/opencv.hpp>

#include "image_dataset.hpp"

#include <iostream>

int main() {
    std::string input_folder_path = "../images/";
    std::string output_folder_path = "../output_images/gaussian_kernel/";
    for (const auto& file_name : noisy_image_filenames()) {
        cv::Mat img = cv::imread(input_folder_path + file_name, cv::IMREAD_COLOR);
        if (img.empty()) {
            std::cout << "Warning: Could not read image: " << file_name << std::endl;
            continue;
        }
        cv::Mat filtered_3x3, filtered_5x5;
        cv::GaussianBlur(img, filtered_3x3, cv::Size(3, 3), 0, 0);
        cv::GaussianBlur(img, filtered_5x5, cv::Size(5, 5), 0, 0);
        cv::Mat bilateral_filtered;
        cv::bilateralFilter(img, bilateral_filtered, 9, 75, 75);
        std::vector<cv::Mat> images = {img, filtered_3x3, filtered_5x5, bilateral_filtered};
        cv::Mat combined_image;
        cv::hconcat(images, combined_image);
        cv::imwrite(output_folder_path + file_name, combined_image);
    }
    return 0;
}