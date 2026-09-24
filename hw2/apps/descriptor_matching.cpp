#include <opencv2/opencv.hpp>

#include "hw2/anms.hpp"
#include "hw2/benchmark.hpp"
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

namespace {

std::vector<cv::KeyPoint> detect_keypoints(const cv::Mat& image, int keep = 500) {
    std::vector<cv::KeyPoint> keypoints;
    detect_dog(image, keypoints);
    return normalize_keypoints(adaptive_non_maximal_suppression(keypoints, keep));
}

}  // namespace

int main() {
    const std::string output_folder = kOutputFolder + "descriptor_matching/";
    std::filesystem::create_directories(output_folder);

    std::ofstream report(output_folder + "results.csv");
    report << "pair,descriptor,tp,fp,fn,tn,precision,recall,accuracy,describe_ms,match_ms\n";

    const float ratio = 0.8f; 
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

        const std::vector<cv::KeyPoint> base1 = detect_keypoints(reference);
        const std::vector<cv::KeyPoint> base2 = detect_keypoints(transformed);
        std::cout << "\n " << pair.name << "\n";

        for (const auto& descriptor : all_descriptors()) {
            std::vector<cv::KeyPoint> keypoints1 = base1;
            std::vector<cv::KeyPoint> keypoints2 = base2;
            cv::Mat descriptors1, descriptors2;

            const double describe_ms = measure_ms(
                [&] {
                    std::vector<cv::KeyPoint> scratch = base1;
                    cv::Mat out;
                    descriptor.compute(reference, scratch, out);
                },
                3);
            descriptor.compute(reference, keypoints1, descriptors1);
            descriptor.compute(transformed, keypoints2, descriptors2);

            if (descriptors1.empty() || descriptors2.empty()) {
                std::cout << "Warning: " << descriptor.name << " produced no descriptors\n";
                continue;
            }

            std::vector<std::vector<cv::DMatch>> knn_matches;
            const double match_ms =
                measure_ms([&] { 
                    knn_matches = 
                    knn_match(
                        descriptors1, 
                        descriptors2, 
                        descriptor.norm_type
                    ); 
                }, 
        3);

            const ConfusionMatrix confusion =
                evaluate_matches(keypoints1, keypoints2, knn_matches, homography, ratio);

            std::cout << "\n " << descriptor.name << " (describe " << describe_ms << " ms, match "
                      << match_ms << " ms)\n"
                      << format_confusion_matrix(confusion) << "\n";

            report << pair.name << "," << descriptor.name << "," << confusion.true_positive << ","
                   << confusion.false_positive << "," << confusion.false_negative << ","
                   << confusion.true_negative << "," << confusion.precision() << ","
                   << confusion.recall() << "," << confusion.accuracy() << "," << describe_ms << ","
                   << match_ms << "\n";

            cv::Mat drawn;
            cv::drawMatches(reference, keypoints1, transformed, keypoints2,
                            filter_matches(knn_matches, ratio), drawn);
            cv::imwrite(output_folder + pair.name + "_" + descriptor.name + ".jpg", drawn);
        }
    }

    std::cout << "\nWrote " << output_folder << "results.csv" << std::endl;
    return 0;
}
