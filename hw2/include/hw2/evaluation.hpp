#pragma once

#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

namespace hw2 {

inline constexpr double kMatchTolerance = 3.0;

double repeatability(const std::vector<cv::KeyPoint>& keypoints1,
                     const std::vector<cv::KeyPoint>& keypoints2,
                     const cv::Mat& homography, cv::Size size2,
                     double tolerance = kMatchTolerance);

struct ConfusionMatrix {
    int true_positive = 0;
    int false_positive = 0;
    int false_negative = 0;
    int true_negative = 0;

    int total() const;
    double true_positive_rate() const;
    double false_positive_rate() const;
    double precision() const;
    double recall() const;
    double accuracy() const;
};

ConfusionMatrix evaluate_matches(
    const std::vector<cv::KeyPoint>& keypoints1,
    const std::vector<cv::KeyPoint>& keypoints2,
    const std::vector<std::vector<cv::DMatch>>& knn_matches,
    const cv::Mat& homography, float ratio,
    double tolerance = kMatchTolerance);

std::string format_confusion_matrix(const ConfusionMatrix& confusion);

struct RocPoint {
    double threshold = 0.0;
    double true_positive_rate = 0.0;
    double false_positive_rate = 0.0;
};

std::vector<RocPoint> compute_roc(
    const std::vector<cv::KeyPoint>& keypoints1,
    const std::vector<cv::KeyPoint>& keypoints2,
    const std::vector<std::vector<cv::DMatch>>& knn_matches,
    const cv::Mat& homography, int steps = 100, float max_ratio = 1.01f);

double area_under_curve(const std::vector<RocPoint>& curve);
RocPoint best_operating_point(const std::vector<RocPoint>& curve);

cv::Mat draw_roc_curves(const std::vector<std::string>& names,
                        const std::vector<std::vector<RocPoint>>& curves);

}  // namespace hw2
