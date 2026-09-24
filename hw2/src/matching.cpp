#include "hw2/matching.hpp"

namespace hw2 {

cv::Mat estimate_ground_truth_homography(const cv::Mat& reference,
                                         const cv::Mat& transformed) {
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints1;
    std::vector<cv::KeyPoint> keypoints2;
    cv::Mat descriptors1;
    cv::Mat descriptors2;
    sift->detectAndCompute(reference, cv::noArray(), keypoints1, descriptors1);
    sift->detectAndCompute(transformed, cv::noArray(), keypoints2, descriptors2);
    if (descriptors1.empty() || descriptors2.empty()) {
        return {};
    }

    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<std::vector<cv::DMatch>> neighbours;
    matcher.knnMatch(descriptors1, descriptors2, neighbours, 2);

    std::vector<cv::Point2f> source;
    std::vector<cv::Point2f> destination;
    for (const auto& pair : neighbours) {
        if (pair.size() == 2 && pair[0].distance < 0.7f * pair[1].distance) {
            source.push_back(keypoints1[pair[0].queryIdx].pt);
            destination.push_back(keypoints2[pair[0].trainIdx].pt);
        }
    }
    if (source.size() < 10) {
        return {};
    }

    cv::Mat inlier_mask;
    cv::Mat homography =
        cv::findHomography(source, destination, cv::RANSAC, 3.0, inlier_mask);
    if (homography.empty() || cv::countNonZero(inlier_mask) < 8 ||
        !cv::checkRange(homography)) {
        return {};
    }
    return homography;
}

std::vector<std::vector<cv::DMatch>> knn_match(const cv::Mat& descriptors1,
                                                const cv::Mat& descriptors2,
                                                int norm_type) {
    std::vector<std::vector<cv::DMatch>> matches;
    if (!descriptors1.empty() && descriptors2.rows >= 2) {
        cv::BFMatcher matcher(norm_type);
        matcher.knnMatch(descriptors1, descriptors2, matches, 2);
    }
    return matches;
}

bool passes_ratio_test(const std::vector<cv::DMatch>& pair, float ratio) {
    if (pair.size() < 2 || ratio <= 0.0f) {
        return false;
    }
    if (pair[1].distance <= 0.0f) {
        return ratio > 1.0f && pair[0].distance <= 0.0f;
    }
    return pair[0].distance < ratio * pair[1].distance;
}

std::vector<cv::DMatch> filter_matches(
    const std::vector<std::vector<cv::DMatch>>& knn_matches, float ratio) {
    std::vector<cv::DMatch> matches;
    for (const auto& pair : knn_matches) {
        if (passes_ratio_test(pair, ratio)) {
            matches.push_back(pair[0]);
        }
    }
    return matches;
}

cv::Point2f project_point(const cv::Mat& homography,
                          const cv::Point2f& point) {
    std::vector<cv::Point2f> input{point};
    std::vector<cv::Point2f> output;
    cv::perspectiveTransform(input, output, homography);
    return output.front();
}

bool inside_image(const cv::Point2f& point, cv::Size size) {
    return point.x >= 0.0f && point.y >= 0.0f && point.x < size.width &&
           point.y < size.height;
}

}  // namespace hw2
