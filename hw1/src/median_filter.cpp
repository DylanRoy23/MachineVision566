#include <opencv2/opencv.hpp>

#include "image_dataset.hpp"

#include <iostream>

int main() {
    std::string input_folder_path = "../images/";
    std::string output_folder_path = "../output_images/median_filter/";
    int kernel_size = 3;
    int kernel_size_5 = 5;
    for (const auto& file_name : noisy_image_filenames()) {
        cv::Mat img = cv::imread(input_folder_path + file_name, cv::IMREAD_COLOR);
        if (img.empty()) {
            std::cout << "Warning: Could not read image: " << file_name << std::endl;
            continue;
        }
        cv::Mat filtered_3x3, filtered_5x5;
        cv::medianBlur(img, filtered_3x3, kernel_size);
        cv::medianBlur(img, filtered_5x5, kernel_size_5);
        cv::imwrite(output_folder_path + file_name, filtered_3x3);
        cv::imwrite(output_folder_path + file_name, filtered_5x5);
        std::vector<cv::Mat> images = {img, filtered_3x3, filtered_5x5};
        cv::Mat combined_image;
        cv::hconcat(images, combined_image);
        cv::imwrite(output_folder_path + file_name, combined_image);
    }
    return 0;
}