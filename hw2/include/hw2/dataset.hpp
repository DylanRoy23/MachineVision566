#pragma once

#include <string>
#include <vector>

namespace hw2 {

extern const std::string kImageFolder;
extern const std::string kOutputFolder;

struct ImagePair {
    std::string name;
    std::string reference;
    std::string transformed;
};

const std::vector<ImagePair>& feature_image_pairs();

}  // namespace hw2
