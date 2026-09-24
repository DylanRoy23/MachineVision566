#pragma once

#include <opencv2/opencv.hpp>

#include <functional>
#include <string>
#include <vector>

namespace hw2 {

struct NamedDetector {
    std::string name;
    std::function<void(const cv::Mat&, std::vector<cv::KeyPoint>&)> detect;
};

void detect_dog(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints);
void detect_harris(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints);
void detect_shi_tomasi(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints);
void detect_fast(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints);
const std::vector<NamedDetector>& all_detectors();

}  // namespace hw2
