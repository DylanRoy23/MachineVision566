#include <opencv2/opencv.hpp>

#include "hw2/benchmark.hpp"
#include "hw2/dataset.hpp"
#include "hw2/detectors.hpp"
#include "hw2/evaluation.hpp"
#include "hw2/matching.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace hw2;

int main() {
    const std::string output_folder = kOutputFolder + "detector_comparison/";
    std::filesystem::create_directories(output_folder);

    std::ofstream report(output_folder + "results.csv");
    report << "pair,detector,keypoints_reference,keypoints_transformed,repeatability,detect_ms\n";

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
            std::cout << "Warning: Could not estimate homography for pair: " << pair.name
                      << " (too little overlap between the two views?)" << std::endl;
            continue;
        }

        std::cout << "\n " << pair.name << "\n";
        std::cout << std::setw(12) << "detector" << std::setw(12) << "features" << std::setw(16)
                  << "repeatability" << std::setw(12) << "time (ms)" << "\n";

        for (const auto& detector : all_detectors()) {
            std::vector<cv::KeyPoint> keypoints1, keypoints2;
            detector.detect(reference, keypoints1);
            detector.detect(transformed, keypoints2);

            const double elapsed_ms = measure_ms([&] {
                std::vector<cv::KeyPoint> scratch;
                detector.detect(reference, scratch);
            });

            const double score =
                repeatability(keypoints1, keypoints2, homography, transformed.size());

            std::cout << std::setw(12) << detector.name << std::setw(12) << keypoints1.size()
                      << std::setw(16) << score << std::setw(12) << elapsed_ms << "\n";

            report << pair.name << "," << detector.name << "," << keypoints1.size() << ","
                   << keypoints2.size() << "," << score << "," << elapsed_ms << "\n";

            cv::Mat annotated;
            cv::drawKeypoints(reference, keypoints1, annotated, cv::Scalar(0, 255, 0),
                              cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            cv::imwrite(output_folder + pair.name + "_" + detector.name + ".jpg", annotated);
        }
    }

    std::cout << "\nWrote " << output_folder << "results.csv" << std::endl;
    return 0;
}
