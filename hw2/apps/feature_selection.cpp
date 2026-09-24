#include <opencv2/opencv.hpp>

#include "hw2/anms.hpp"
#include "hw2/benchmark.hpp"
#include "hw2/dataset.hpp"
#include "hw2/detectors.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace hw2;

int main() {
    const std::string output_folder = kOutputFolder + "feature_selection/";
    std::filesystem::create_directories(output_folder);

    const int keep = 400;

    std::ofstream report(output_folder + "results.csv");
    report << "image,method,keypoints,grid_coverage,selection_ms\n";

    std::cout << std::fixed << std::setprecision(3);

    std::vector<std::string> file_names;
    for (const auto& pair : feature_image_pairs()) {
        for (const auto& name : {pair.reference, pair.transformed}) {
            if (std::find(file_names.begin(), file_names.end(), name) == file_names.end()) {
                file_names.push_back(name);
            }
        }
    }

    for (const auto& file_name : file_names) {
        const cv::Mat image = cv::imread(kImageFolder + file_name, cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            std::cout << "Warning: Could not read image: " << file_name << std::endl;
            continue;
        }
        std::string stem = file_name.substr(0, file_name.find_last_of('.'));
        std::replace(stem.begin(), stem.end(), '/', '_');

        std::vector<cv::KeyPoint> keypoints;
        detect_dog(image, keypoints);
        std::cout << "\n " << file_name << " (" << keypoints.size() << " detected)\n";
        std::cout << std::setw(16) << "method" << std::setw(12) << "kept" << std::setw(16)
                  << "coverage" << std::setw(12) << "time (ms)" << "\n";

        std::vector<cv::KeyPoint> strongest, adaptive;
        const double strongest_ms =
            measure_ms([&] { strongest = strongest_keypoints(keypoints, keep); });
        const double adaptive_ms =
            measure_ms([&] { adaptive = adaptive_non_maximal_suppression(keypoints, keep); }, 1);

        const std::vector<std::pair<std::string, std::vector<cv::KeyPoint>>> results = {
            {"strongest-N", strongest}, {"ANMS", adaptive}};
        const std::vector<double> timings = {strongest_ms, adaptive_ms};

        for (size_t i = 0; i < results.size(); ++i) {
            const auto& [name, selected] = results[i];
            const double coverage = grid_coverage(selected, image.size());

            std::cout << std::setw(16) << name << std::setw(12) << selected.size() << std::setw(16)
                      << coverage << std::setw(12) << timings[i] << "\n";
            report << file_name << "," << name << "," << selected.size() << "," << coverage
                   << "," << timings[i] << "\n";

            cv::Mat annotated;
            cv::drawKeypoints(image, selected, annotated, cv::Scalar(0, 255, 0));
            cv::imwrite(output_folder + stem + "_" + name + ".jpg", annotated);
        }

        cv::Mat left, right, combined;
        cv::drawKeypoints(image, strongest, left, cv::Scalar(0, 0, 255));
        cv::drawKeypoints(image, adaptive, right, cv::Scalar(0, 255, 0));
        cv::hconcat(std::vector<cv::Mat>{left, right}, combined);
        cv::imwrite(output_folder + stem + "_comparison.jpg", combined);
    }

    std::cout << "\nWrote " << output_folder << "results.csv" << std::endl;
    return 0;
}
