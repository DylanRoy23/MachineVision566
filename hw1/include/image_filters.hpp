#pragma once

#include <opencv2/core.hpp>

#include <algorithm>
#include <vector>

inline void n_rank_filter(const cv::Mat& image, cv::Mat& filtered, int kernel_size, int rank) {
    const int radius = kernel_size / 2;
    const int channels = image.channels();
    filtered.create(image.size(), image.type());

    std::vector<uchar> pixel_values;
    pixel_values.reserve(kernel_size * kernel_size);

    for (int row = 0; row < image.rows; ++row) {
        for (int col = 0; col < image.cols; ++col) {
            uchar* output_pixel = filtered.ptr<uchar>(row) + col * channels;

            for (int channel = 0; channel < channels; ++channel) {
                pixel_values.clear();
                for (int y = -radius; y <= radius; ++y) {
                    const int source_row = std::clamp(row + y, 0, image.rows - 1);
                    const uchar* source_pixels = image.ptr<uchar>(source_row);
                    for (int x = -radius; x <= radius; ++x) {
                        const int source_col = std::clamp(col + x, 0, image.cols - 1);
                        pixel_values.push_back(source_pixels[source_col * channels + channel]);
                    }
                }

                std::nth_element(pixel_values.begin(), pixel_values.begin() + rank, pixel_values.end());
                output_pixel[channel] = pixel_values[rank];
            }
        }
    }
}