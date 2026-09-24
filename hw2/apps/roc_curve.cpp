#include <opencv2/opencv.hpp>

#include "hw2/anms.hpp"
#include "hw2/dataset.hpp"
#include "hw2/descriptors.hpp"
#include "hw2/detectors.hpp"
#include "hw2/evaluation.hpp"
#include "hw2/matching.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace hw2;

int main() {
    const std::string output_folder = kOutputFolder + "roc_curve/";
    std::filesystem::create_directories(output_folder);

    std::ofstream report(output_folder + "roc_points.csv");
    report << "pair,descriptor,ratio_threshold,true_positive_rate,false_positive_rate\n";

    std::cout << std::fixed << std::setprecision(3);

    for (const auto& pair : feature_image_pairs()) {
        const cv::Mat reference = cv::imread(kImageFolder + pair.reference, cv::IMREAD_GRAYSCALE);
        const cv::Mat transformed = cv::imread(kImageFolder + pair.transformed, cv::IMREAD_GRAYSCALE);
        if (reference.empty() || transformed.empty()) {
            std::cout << "Warning: Could not read image pair: " << pair.name << std::endl;
            continue;
        }

        const cv::Mat homography = estimate_ground_truth_homography(reference, transformed);
        if (homography.empty()) {
            std::cout << "Warning: Could not estimate homography for pair: " << pair.name << std::endl;
            continue;
        }

        std::vector<cv::KeyPoint> detected1, detected2;
        detect_dog(reference, detected1);
        detect_dog(transformed, detected2);
        const std::vector<cv::KeyPoint> base1 =
            normalize_keypoints(adaptive_non_maximal_suppression(detected1, 500));
        const std::vector<cv::KeyPoint> base2 =
            normalize_keypoints(adaptive_non_maximal_suppression(detected2, 500));

        std::vector<std::string> names;
        std::vector<std::vector<RocPoint>> curves;

        std::cout << "\n " << pair.name << "\n";
        for (const auto& descriptor : all_descriptors()) {
            std::vector<cv::KeyPoint> keypoints1 = base1;
            std::vector<cv::KeyPoint> keypoints2 = base2;
            cv::Mat descriptors1, descriptors2;
            descriptor.compute(reference, keypoints1, descriptors1);
            descriptor.compute(transformed, keypoints2, descriptors2);
            if (descriptors1.empty() || descriptors2.empty()) {
                continue;
            }

            const std::vector<std::vector<cv::DMatch>> knn_matches =
                knn_match(descriptors1, descriptors2, descriptor.norm_type);
            const std::vector<RocPoint> curve =
                compute_roc(keypoints1, keypoints2, knn_matches, homography);

            const RocPoint best = best_operating_point(curve);
            std::cout << std::setw(8) << descriptor.name << "  AUC " << area_under_curve(curve)
                      << "  best ratio " << best.threshold << " (TPR " << best.true_positive_rate
                      << ", FPR " << best.false_positive_rate << ")\n";

            for (const auto& point : curve) {
                report << pair.name << "," << descriptor.name << "," << point.threshold << ","
                       << point.true_positive_rate << "," << point.false_positive_rate << "\n";
            }

            names.push_back(descriptor.name);
            curves.push_back(curve);
        }

        if (!curves.empty()) {
            cv::imwrite(output_folder + pair.name + "_roc.png", draw_roc_curves(names, curves));
        }
    }

    std::cout << "\nWrote " << output_folder << "roc_points.csv" << std::endl;
    return 0;
}
