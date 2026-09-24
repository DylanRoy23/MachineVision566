#pragma once

#include <functional>

namespace hw2 {

double measure_ms(const std::function<void()>& work, int runs = 5);

}  // namespace hw2
