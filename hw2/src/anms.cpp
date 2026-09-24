#include "hw2/anms.hpp"

#include <algorithm>
#include <limits>
#include <utility>

namespace hw2 {

std::vector<cv::KeyPoint> adaptive_non_maximal_suppression(
    const std::vector<cv::KeyPoint>& keypoints, int num_to_keep,
    float robust_ratio) {
    if (num_to_keep <= 0) {
        return {};
    }
    if (keypoints.size() <= static_cast<size_t>(num_to_keep)) {
        return keypoints;
    }

    std::vector<cv::KeyPoint> sorted = keypoints;
    std::sort(sorted.begin(), sorted.end(),
              [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
                  return a.response > b.response;
              });

    std::vector<std::pair<float, size_t>> radii;
    radii.reserve(sorted.size());
    for (size_t i = 0; i < sorted.size(); ++i) {
        float radius = std::numeric_limits<float>::max();
        for (size_t j = 0; j < i; ++j) {
            if (sorted[j].response * robust_ratio > sorted[i].response) {
                radius = std::min(
                    radius,
                    static_cast<float>(cv::norm(sorted[i].pt - sorted[j].pt)));
            }
        }
        radii.emplace_back(radius, i);
    }

    std::sort(radii.begin(), radii.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    std::vector<cv::KeyPoint> selected;
    selected.reserve(num_to_keep);
    for (int i = 0; i < num_to_keep; ++i) {
        selected.push_back(sorted[radii[i].second]);
    }
    return selected;
}

std::vector<cv::KeyPoint> strongest_keypoints(
    const std::vector<cv::KeyPoint>& keypoints, int num_to_keep) {
    if (num_to_keep <= 0) {
        return {};
    }
    if (keypoints.size() <= static_cast<size_t>(num_to_keep)) {
        return keypoints;
    }

    std::vector<cv::KeyPoint> sorted = keypoints;
    std::sort(sorted.begin(), sorted.end(),
              [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
                  return a.response > b.response;
              });
    sorted.resize(num_to_keep);
    return sorted;
}

double grid_coverage(const std::vector<cv::KeyPoint>& keypoints,
                     cv::Size image_size, int grid) {
    if (grid <= 0 || image_size.width <= 0 || image_size.height <= 0) {
        return 0.0;
    }

    std::vector<bool> occupied(grid * grid, false);
    for (const auto& keypoint : keypoints) {
        const int col = std::clamp(
            static_cast<int>(keypoint.pt.x * grid / image_size.width), 0,
            grid - 1);
        const int row = std::clamp(
            static_cast<int>(keypoint.pt.y * grid / image_size.height), 0,
            grid - 1);
        occupied[row * grid + col] = true;
    }
    return static_cast<double>(
               std::count(occupied.begin(), occupied.end(), true)) /
           occupied.size();
}

}  // namespace hw2
