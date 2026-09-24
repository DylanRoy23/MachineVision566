#include "hw2/evaluation.hpp"

#include "hw2/matching.hpp"

#include <algorithm>
#include <iomanip>
#include <sstream>

namespace hw2 {

double repeatability(const std::vector<cv::KeyPoint>& keypoints1,
                     const std::vector<cv::KeyPoint>& keypoints2,
                     const cv::Mat& homography, cv::Size size2,
                     double tolerance) {
    if (keypoints1.empty() || keypoints2.empty() || homography.empty()) {
        return 0.0;
    }

    struct Candidate {
        double distance;
        size_t reference_index;
        size_t transformed_index;
    };

    int visible = 0;
    std::vector<Candidate> candidates;
    for (size_t i = 0; i < keypoints1.size(); ++i) {
        const cv::Point2f projected = project_point(homography, keypoints1[i].pt);
        if (!inside_image(projected, size2)) {
            continue;
        }
        ++visible;
        for (size_t j = 0; j < keypoints2.size(); ++j) {
            const double distance = cv::norm(keypoints2[j].pt - projected);
            if (distance <= tolerance) {
                candidates.push_back({distance, i, j});
            }
        }
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) {
                  return a.distance < b.distance;
              });
    std::vector<char> used_reference(keypoints1.size(), 0);
    std::vector<char> used_transformed(keypoints2.size(), 0);
    int repeated = 0;
    for (const auto& candidate : candidates) {
        if (!used_reference[candidate.reference_index] &&
            !used_transformed[candidate.transformed_index]) {
            used_reference[candidate.reference_index] = 1;
            used_transformed[candidate.transformed_index] = 1;
            ++repeated;
        }
    }
    return visible == 0 ? 0.0 : static_cast<double>(repeated) / visible;
}

int ConfusionMatrix::total() const {
    return true_positive + false_positive + false_negative + true_negative;
}

double ConfusionMatrix::true_positive_rate() const {
    const int positives = true_positive + false_negative;
    return positives == 0 ? 0.0 : static_cast<double>(true_positive) / positives;
}

double ConfusionMatrix::false_positive_rate() const {
    const int negatives = false_positive + true_negative;
    return negatives == 0 ? 0.0 : static_cast<double>(false_positive) / negatives;
}

double ConfusionMatrix::precision() const {
    const int predicted = true_positive + false_positive;
    return predicted == 0 ? 0.0 : static_cast<double>(true_positive) / predicted;
}

double ConfusionMatrix::recall() const { return true_positive_rate(); }

double ConfusionMatrix::accuracy() const {
    return total() == 0
               ? 0.0
               : static_cast<double>(true_positive + true_negative) / total();
}

ConfusionMatrix evaluate_matches(
    const std::vector<cv::KeyPoint>& keypoints1,
    const std::vector<cv::KeyPoint>& keypoints2,
    const std::vector<std::vector<cv::DMatch>>& knn_matches,
    const cv::Mat& homography, float ratio, double tolerance) {
    ConfusionMatrix confusion;
    if (homography.empty()) {
        return confusion;
    }

    for (const auto& neighbours : knn_matches) {
        if (neighbours.empty()) {
            continue;
        }
        const cv::DMatch& nearest = neighbours[0];
        if (nearest.queryIdx < 0 || nearest.trainIdx < 0 ||
            nearest.queryIdx >= static_cast<int>(keypoints1.size()) ||
            nearest.trainIdx >= static_cast<int>(keypoints2.size())) {
            continue;
        }

        const cv::Point2f projected =
            project_point(homography, keypoints1[nearest.queryIdx].pt);
        const bool correct =
            cv::norm(keypoints2[nearest.trainIdx].pt - projected) <= tolerance;
        const bool accepted = passes_ratio_test(neighbours, ratio);

        if (accepted && correct) {
            ++confusion.true_positive;
        } else if (accepted) {
            ++confusion.false_positive;
        } else if (correct) {
            ++confusion.false_negative;
        } else {
            ++confusion.true_negative;
        }
    }
    return confusion;
}

std::string format_confusion_matrix(const ConfusionMatrix& confusion) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(3);
    out << "                 predicted match  predicted no-match\n";
    out << "  correspondence   " << std::setw(10) << confusion.true_positive
        << std::setw(20) << confusion.false_negative << '\n';
    out << "  none             " << std::setw(10) << confusion.false_positive
        << std::setw(20) << confusion.true_negative << '\n';
    out << "  precision " << confusion.precision() << "  recall "
        << confusion.recall() << "  accuracy " << confusion.accuracy();
    return out.str();
}

std::vector<RocPoint> compute_roc(
    const std::vector<cv::KeyPoint>& keypoints1,
    const std::vector<cv::KeyPoint>& keypoints2,
    const std::vector<std::vector<cv::DMatch>>& knn_matches,
    const cv::Mat& homography, int steps, float max_ratio) {
    std::vector<RocPoint> curve;
    curve.reserve(steps + 1);
    for (int step = 0; step <= steps; ++step) {
        const float ratio = max_ratio * static_cast<float>(step) / steps;
        const ConfusionMatrix confusion = evaluate_matches(
            keypoints1, keypoints2, knn_matches, homography, ratio);
        curve.push_back({ratio, confusion.true_positive_rate(),
                         confusion.false_positive_rate()});
    }
    return curve;
}

double area_under_curve(const std::vector<RocPoint>& curve) {
    double area = 0.0;
    for (size_t i = 1; i < curve.size(); ++i) {
        const double width =
            curve[i].false_positive_rate - curve[i - 1].false_positive_rate;
        const double height =
            (curve[i].true_positive_rate + curve[i - 1].true_positive_rate) /
            2.0;
        area += width * height;
    }
    return area;
}

RocPoint best_operating_point(const std::vector<RocPoint>& curve) {
    RocPoint best;
    double best_score = -1.0;
    for (const auto& point : curve) {
        if (point.threshold > 1.0) {
            continue;
        }
        const double score = point.true_positive_rate - point.false_positive_rate;
        if (score > best_score) {
            best_score = score;
            best = point;
        }
    }
    return best;
}

cv::Mat draw_roc_curves(const std::vector<std::string>& names,
                        const std::vector<std::vector<RocPoint>>& curves) {
    constexpr int size = 500;
    constexpr int pad = 60;
    cv::Mat plot(size + pad * 2, size + pad * 2, CV_8UC3,
                 cv::Scalar(255, 255, 255));
    const auto to_pixel = [&](double fpr, double tpr) {
        return cv::Point(pad + static_cast<int>(fpr * size),
                         pad + size - static_cast<int>(tpr * size));
    };

    cv::rectangle(plot, to_pixel(0, 1), to_pixel(1, 0), cv::Scalar(0, 0, 0));
    cv::line(plot, to_pixel(0, 0), to_pixel(1, 1),
             cv::Scalar(180, 180, 180), 1, cv::LINE_AA);
    cv::putText(plot, "false positive rate",
                cv::Point(pad + size / 2 - 70, pad + size + 40),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1,
                cv::LINE_AA);
    cv::putText(plot, "true positive rate", cv::Point(5, pad - 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1,
                cv::LINE_AA);

    const std::vector<cv::Scalar> colors = {
        {200, 60, 60}, {60, 150, 60}, {60, 60, 200}, {180, 120, 40}};
    for (size_t c = 0; c < curves.size(); ++c) {
        const cv::Scalar color = colors[c % colors.size()];
        for (size_t i = 1; i < curves[c].size(); ++i) {
            cv::line(plot,
                     to_pixel(curves[c][i - 1].false_positive_rate,
                              curves[c][i - 1].true_positive_rate),
                     to_pixel(curves[c][i].false_positive_rate,
                              curves[c][i].true_positive_rate),
                     color, 2, cv::LINE_AA);
        }

        std::ostringstream label;
        label << std::fixed << std::setprecision(3) << names[c] << "  AUC "
              << area_under_curve(curves[c]);
        cv::putText(plot, label.str(),
                    cv::Point(pad + size / 2 + 20,
                              pad + size - 70 + static_cast<int>(c) * 22),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv::LINE_AA);
    }
    return plot;
}

}  // namespace hw2
