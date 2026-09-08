#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    std::string input_folder_path = "../images/";
    std::string output_folder_path = "../output_images/thresholding/";
    std::vector<std::string> file_names = {
        "coins1.jpg", 
        "coins2.jpg", 
        "screws.jpeg"
    };
    for (const auto& file_name : file_names) {
        cv::Mat img = cv::imread(input_folder_path + file_name, cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            std::cout << "Warning: Could not read image: " << file_name << std::endl;
            continue;
        }
        cv::Mat thresh;\
        cv::Mat adaptive_thresh;
        cv::threshold(img, thresh, 127, 255, cv::THRESH_BINARY);
        cv::adaptiveThreshold(
            img, 
            adaptive_thresh,
            255, 
            cv::ADAPTIVE_THRESH_MEAN_C,
            cv::THRESH_BINARY, 
            11, 
            2
        );
        cv::imwrite(output_folder_path + file_name, thresh);
        cv::imwrite(output_folder_path + file_name, adaptive_thresh);
        std::vector<cv::Mat> images = {img, thresh, adaptive_thresh};
        cv::Mat combined_image;
        cv::hconcat(images, combined_image);
        cv::imwrite(output_folder_path + file_name, combined_image);
    }
    return 0;
}