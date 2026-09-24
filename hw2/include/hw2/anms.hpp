#pragma once

#include <opencv2/opencv.hpp>

#include <vector>

namespace hw2 {

std::vector<cv::KeyPoint> adaptive_non_maximal_suppression(
    const std::vector<cv::KeyPoint>& keypoints, int num_to_keep,
    float robust_ratio = 0.9f);

std::vector<cv::KeyPoint> strongest_keypoints(
    const std::vector<cv::KeyPoint>& keypoints, int num_to_keep);

double grid_coverage(const std::vector<cv::KeyPoint>& keypoints,
                     cv::Size image_size, int grid = 10);

}  // namespace hw2
