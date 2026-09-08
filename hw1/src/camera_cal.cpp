#include <opencv2/opencv.hpp>
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <random>
#include <vector>

namespace fs = std::filesystem;

std::vector<std::string> getImages(const fs::path& folder) {
    std::vector<std::string> images;
    for (const auto& file : fs::directory_iterator(folder)) {
        if (file.path().extension() == ".jpg")
            images.push_back(file.path().string());
    }
    std::sort(images.begin(), images.end());
    return images;
}

void calibrateDataset(const std::string& name,
                      const std::vector<std::string>& files) {
    const cv::Size boardSize(7, 10); // internal checkerboard corners
    std::vector<cv::Point3f> boardPoints;
    for (int row = 0; row < boardSize.height; row++)
        for (int col = 0; col < boardSize.width; col++)
            boardPoints.emplace_back(col, row, 0);

    std::vector<std::vector<cv::Point2f>> imagePoints;
    cv::Size imageSize;

    // Find 25 usable checkerboard images.
    for (const std::string& file : files) {
        cv::Mat image = cv::imread(file);
        if (image.empty()) continue;

        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        std::vector<cv::Point2f> corners;
        if (cv::findChessboardCorners(gray, boardSize, corners)) {
            cv::cornerSubPix(gray, corners, {11, 11}, {-1, -1},
                {cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.001});
            imagePoints.push_back(corners);
            imageSize = image.size();
            if (imagePoints.size() == 500) break;
        }
    }

    if (imagePoints.size() < 500) {
        std::cerr << name << " does not have 25 usable images.\n";
        return;
    }

    std::cout << "\n" << name << "\n";
    for (int count : {2, 5, 10, 25, 50, 250, 500}) {
        std::vector<std::vector<cv::Point2f>> points(
            imagePoints.begin(), imagePoints.begin() + count);
        std::vector<std::vector<cv::Point3f>> objects(count, boardPoints);
        cv::Mat cameraMatrix, distortion;
        std::vector<cv::Mat> rotations, translations;

        double error = cv::calibrateCamera(objects, points, imageSize,
            cameraMatrix, distortion, rotations, translations);

        std::cout << "\nImages: " << count
                  << "\nRMS error: " << error
                  << "\nCamera matrix:\n" << cameraMatrix
                  << "\nDistortion: " << distortion << "\n";
    }
}

int main(int argc, char** argv) {
    fs::path folder = argc > 1 ? argv[1] : "../cal_images";
    auto set1 = getImages(folder / "cal_image_set_1");
    auto set2 = getImages(folder / "cal_image_set_2");

    // Third dataset, this is a reproducible sample of the two other dataseets
    auto set3 = set1;
    set3.insert(set3.end(), set2.begin(), set2.end());
    std::shuffle(set3.begin(), set3.end(), std::mt19937(42));

    calibrateDataset("Dataset 1", set1);
    calibrateDataset("Dataset 2", set2);
    calibrateDataset("Dataset 3 (random sample)", set3);
    return 0;
}
