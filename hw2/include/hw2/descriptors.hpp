#pragma once

#include <opencv2/opencv.hpp>

#include <functional>
#include <random>
#include <string>
#include <vector>

namespace hw2 {

class BriefDescriptor {
public:
    explicit BriefDescriptor(int bytes = 32, int patch_size = 32);

    void compute(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints,
                 cv::Mat& descriptors) const;

private:
    static cv::Point sample_offset(std::mt19937& generator,
                                   std::normal_distribution<float>& spread,
                                   int limit);

    int bytes_;
    int patch_size_;
    std::vector<cv::Point> point_a_;
    std::vector<cv::Point> point_b_;
};

struct NamedDescriptor {
    std::string name;
    int norm_type;
    std::function<void(const cv::Mat&, std::vector<cv::KeyPoint>&, cv::Mat&)> compute;
};

std::vector<cv::KeyPoint> normalize_keypoints(
    std::vector<cv::KeyPoint> keypoints, float default_size = 31.0f);
const std::vector<NamedDescriptor>& all_descriptors();

}  // namespace hw2
