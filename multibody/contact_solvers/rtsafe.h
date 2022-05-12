#pragma once

#include <functional>
#include <tuple>
#include <utility>

#include "drake/common/drake_assert.h"
#include "drake/common/drake_throw.h"
#include "drake/common/text_logging.h"

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {

struct NewtonWithBisectionFallbackParams {
  double abs_tolerance{std::numeric_limits<double>::epsilon()};
  int max_iterations{100};
  bool verify_interval{true};
};

/*

*/
std::pair<double, int> NewtonWithBisectionFallback(
    const std::function<std::pair<double, double>(double)>& function,
    double x_lower, double x_upper, double x_guess,
    const NewtonWithBisectionFallbackParams& params) {
  using std::abs;
  using std::max;
  using std::min;
  using std::swap;
  DRAKE_THROW_UNLESS(params.abs_tolerance > 0);
  DRAKE_THROW_UNLESS(params.max_iterations > 0);

  // These checks verify there is an appropriate bracket around the root,
  // though at the expense of additional evaluations.
  if (params.verify_interval) {
    DRAKE_THROW_UNLESS(x_upper > x_lower);
    DRAKE_THROW_UNLESS(x_lower <= x_guess && x_guess <= x_upper);

    const auto [f_lower, df_lower] = function(x_lower);
    if (f_lower == 0) return std::make_pair(x_lower, 1);

    const auto [f_upper, df_upper] = function(x_upper);
    if (f_upper == 0) return std::make_pair(x_upper, 2);

    DRAKE_THROW_UNLESS(f_lower * f_upper <= 0);

    // Re-orient the search so that f(x_lower) < 0.
    if (f_lower > 0) swap(x_lower, x_upper);
  }

  double root = x_guess;  // Initialize to user supplied guess.
  double dx_previous = (x_upper - x_lower);
  double minus_dx = dx_previous;
  auto [f, df] = function(root);
  for (int num_evaluations = 1; num_evaluations <= params.max_iterations;
       ++num_evaluations) {

    if (((root - x_upper) * df - f) * ((root - x_lower) * df - f) > 0.0 ||
        abs(2.0 * f) > abs(dx_previous * df)) {
      // Bisection: Newton's method would either take us out of bounds or is not
      // reducing the size of the bracket fast enough.          
      dx_previous = minus_dx;
      minus_dx = 0.5 * (x_upper - x_lower);
      root = x_lower + minus_dx;
      DRAKE_LOGGER_DEBUG(
          "Bisect. k = {:d}. x = {:12.6g}. [x_lower, x_upper] = [{:12.8g}, "
          "{:12.8g}].",
          num_evaluations, root, x_lower, x_upper);
      if (x_lower == root) {
        return std::make_pair(root, num_evaluations);
      }
    } else {
      // Newton method.
      dx_previous = minus_dx;
      minus_dx = f / df;
      double previous_root = root;
      root -= minus_dx;
      DRAKE_LOGGER_DEBUG(
          "Newton. k = {:d}. x = {:12.6g}. [x_lower, x_upper] = [{:12.8g}, "
          "{:12.8g}].",
          num_evaluations, root, x_lower, x_upper);
      if (previous_root == root) {
        // If previous_root equals root "after" the update, it means that minus_dx is
        // small compared to root's value, in floating point precision.
        return std::make_pair(root, num_evaluations);
      }
    }

    // Return if minus_dx is within the specified absolute tolerance.
    if (abs(minus_dx) < params.abs_tolerance) {      
      return std::make_pair(root, num_evaluations);
    }

    // The one single evaluation of the function and its derivatives per
    // iteration.
    std::tie(f, df) = function(root);

    // Update the bracket around root.
    if (f < 0.0) {
      x_lower = root;
    } else {
      x_upper = root;
    }
  }

  // If here, then NewtonWithBisectionFallback did not converge.
  throw std::runtime_error(
      fmt::format("NewtonWithBisectionFallback did not converge.\n"
                  "|x - x_prev| = {}. |x_upper-x_lower| = {}",
                  abs(minus_dx), abs(x_upper - x_lower)));
};

}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake
