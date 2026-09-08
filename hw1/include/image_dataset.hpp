#pragma once

#include <string>
#include <vector>

inline const std::vector<std::string>& noisy_image_filenames() {
    static const std::vector<std::string> filenames = {
        "camera_man_w_noise.jpg",
        "SaltAndPepperNoise.jpg",
        "GaussianNoise.jpg",
        "UniformNoise.jpg",
    };
    return filenames;
}