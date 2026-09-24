#pragma once

#include <opencv2/opencv.hpp>

#include <vector>

namespace hw2 {

cv::Mat estimate_ground_truth_homography(const cv::Mat& reference,
                                         const cv::Mat& transformed);

std::vector<std::vector<cv::DMatch>> knn_match(const cv::Mat& descriptors1,
                                                const cv::Mat& descriptors2,
                                                int norm_type);

bool passes_ratio_test(const std::vector<cv::DMatch>& pair, float ratio);

std::vector<cv::DMatch> filter_matches(
    const std::vector<std::vector<cv::DMatch>>& knn_matches, float ratio);

cv::Point2f project_point(const cv::Mat& homography, const cv::Point2f& point);
bool inside_image(const cv::Point2f& point, cv::Size size);

}  // namespace hw2
