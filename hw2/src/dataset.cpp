#include "hw2/dataset.hpp"

namespace hw2 {

const std::string kImageFolder = "../images/";
const std::string kOutputFolder = "../output_images/";

const std::vector<ImagePair>& feature_image_pairs() {
    static const std::vector<ImagePair> pairs = {
        {"scale", "hw2_scale_images/frame_100.jpg", "hw2_scale_images/frame_150.jpg"},
        {"rotation", "hw2_rot_images/frame_100.jpg", "hw2_rot_images/frame_150.jpg"},
        {"translation", "hw2_translation_images/frame_130.jpg",
         "hw2_translation_images/frame_160.jpg"},
    };
    return pairs;
}

}  // namespace hw2
