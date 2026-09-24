#include "hw2/benchmark.hpp"

#include <chrono>

namespace hw2 {

double measure_ms(const std::function<void()>& work, int runs) {
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < runs; ++i) {
        work();
    }
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double, std::milli> elapsed = end - start;
    return elapsed.count() / runs;
}

}  // namespace hw2
