#include "hw2/detectors.hpp"

namespace hw2 {
namespace {

constexpr int kMaxCorners = 2000;

}  // namespace

void detect_dog(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints) {
    static cv::Ptr<cv::SIFT> detector = cv::SIFT::create();
    detector->detect(image, keypoints);
}

void detect_harris(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints) {
    static cv::Ptr<cv::GFTTDetector> detector =
        cv::GFTTDetector::create(kMaxCorners, 0.01, 10.0, 3, true, 0.04);
    detector->detect(image, keypoints);
}

void detect_shi_tomasi(const cv::Mat& image,
                       std::vector<cv::KeyPoint>& keypoints) {
    static cv::Ptr<cv::GFTTDetector> detector =
        cv::GFTTDetector::create(kMaxCorners, 0.01, 10.0, 3, false);
    detector->detect(image, keypoints);
}

void detect_fast(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints) {
    static cv::Ptr<cv::FastFeatureDetector> detector =
        cv::FastFeatureDetector::create(
            20, true, cv::FastFeatureDetector::TYPE_9_16);
    detector->detect(image, keypoints);
}

const std::vector<NamedDetector>& all_detectors() {
    static const std::vector<NamedDetector> detectors = {
        {"DoG", detect_dog},
        {"Harris", detect_harris},
        {"Shi-Tomasi", detect_shi_tomasi},
        {"FAST", detect_fast},
    };
    return detectors;
}

}  // namespace hw2
