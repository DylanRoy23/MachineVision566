#include "hw2/descriptors.hpp"

#include <algorithm>

namespace hw2 {

BriefDescriptor::BriefDescriptor(int bytes, int patch_size)
    : bytes_(bytes), patch_size_(patch_size) {
    std::mt19937 generator(42);
    std::normal_distribution<float> spread(0.0f, patch_size_ / 5.0f);
    const int limit = patch_size_ / 2 - 1;

    const int bits = bytes_ * 8;
    point_a_.reserve(bits);
    point_b_.reserve(bits);
    for (int i = 0; i < bits; ++i) {
        point_a_.push_back(sample_offset(generator, spread, limit));
        point_b_.push_back(sample_offset(generator, spread, limit));
    }
}

void BriefDescriptor::compute(const cv::Mat& image,
                              std::vector<cv::KeyPoint>& keypoints,
                              cv::Mat& descriptors) const {
    cv::Mat smoothed;
    cv::GaussianBlur(image, smoothed, cv::Size(9, 9), 2.0, 2.0);

    const int margin = patch_size_ / 2 + 1;
    std::vector<cv::KeyPoint> kept;
    kept.reserve(keypoints.size());
    for (const auto& keypoint : keypoints) {
        const int x = cvRound(keypoint.pt.x);
        const int y = cvRound(keypoint.pt.y);
        if (x >= margin && y >= margin && x < image.cols - margin &&
            y < image.rows - margin) {
            kept.push_back(keypoint);
        }
    }

    descriptors = cv::Mat::zeros(static_cast<int>(kept.size()), bytes_, CV_8U);
    for (size_t i = 0; i < kept.size(); ++i) {
        const int x = cvRound(kept[i].pt.x);
        const int y = cvRound(kept[i].pt.y);
        uchar* row = descriptors.ptr<uchar>(static_cast<int>(i));
        for (size_t bit = 0; bit < point_a_.size(); ++bit) {
            const uchar first = smoothed.at<uchar>(
                y + point_a_[bit].y, x + point_a_[bit].x);
            const uchar second = smoothed.at<uchar>(
                y + point_b_[bit].y, x + point_b_[bit].x);
            if (first < second) {
                row[bit / 8] |= static_cast<uchar>(1U << (bit % 8));
            }
        }
    }
    keypoints = std::move(kept);
}

cv::Point BriefDescriptor::sample_offset(
    std::mt19937& generator, std::normal_distribution<float>& spread,
    int limit) {
    return {
        std::clamp(cvRound(spread(generator)), -limit, limit),
        std::clamp(cvRound(spread(generator)), -limit, limit),
    };
}

std::vector<cv::KeyPoint> normalize_keypoints(
    std::vector<cv::KeyPoint> keypoints, float default_size) {
    for (auto& keypoint : keypoints) {
        keypoint.octave = 0;
        if (keypoint.size <= 0.0f) {
            keypoint.size = default_size;
        }
    }
    return keypoints;
}

const std::vector<NamedDescriptor>& all_descriptors() {
    static const BriefDescriptor brief;
    static const std::vector<NamedDescriptor> descriptors = {
        {"SIFT", cv::NORM_L2,
         [](const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints,
            cv::Mat& out) {
             static cv::Ptr<cv::SIFT> extractor = cv::SIFT::create();
             extractor->compute(image, keypoints, out);
         }},
        {"BRIEF", cv::NORM_HAMMING,
         [](const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints,
            cv::Mat& out) { brief.compute(image, keypoints, out); }},
        {"ORB", cv::NORM_HAMMING,
         [](const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints,
            cv::Mat& out) {
             static cv::Ptr<cv::ORB> extractor = cv::ORB::create();
             extractor->compute(image, keypoints, out);
         }},
    };
    return descriptors;
}

}  // namespace hw2
