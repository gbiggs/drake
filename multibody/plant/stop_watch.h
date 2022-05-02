#pragma once

#include <chrono>

namespace drake {
namespace multibody {
namespace internal {

class StopWatch {
 public:
  StopWatch() { Reset(); }

  // Resets the stopwatch.
  void Reset() { start_ = clock::now(); }

  // Returns the elapsed time, in seconds, since the last
  // call to Reset().
  double Elapsed() {
    clock::time_point end = clock::now();
    return std::chrono::duration<double>(end - start_).count();
  }

 private:
  using clock = std::chrono::steady_clock;
  clock::time_point start_;
};

}  // namespace internal
}  // namespace multibody
}  // namespace drake