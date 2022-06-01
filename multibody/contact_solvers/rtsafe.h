#pragma once

#include <functional>
#include <iostream>
#include <tuple>
#include <utility>

#include "drake/common/drake_assert.h"
#include "drake/common/drake_throw.h"
#include "drake/common/text_logging.h"
#define PRINT_VAR(a) std::cout << #a ": " << a << std::endl;

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
  f(x_lower) < 0
  f(x_upper) > 0
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

  if (x_lower > x_upper) swap(x_lower, x_upper);

  // These checks verify there is an appropriate bracket around the root,
  // though at the expense of additional evaluations.
  auto [f_lower, df_lower] = function(x_lower);
  if (f_lower == 0) return std::make_pair(x_lower, 1);

  auto [f_upper, df_upper] = function(x_upper);
  if (f_upper == 0) return std::make_pair(x_upper, 2);

  // Verify guess is inside the bracket.
  DRAKE_THROW_UNLESS(f_lower * f_upper <= 0);

  double root = x_guess;  // Initialize to user supplied guess.
  double previous_minus_dx = (x_upper - x_lower);
  double minus_dx = previous_minus_dx;
  auto [f, df] = function(root);
  for (int num_evaluations = 1; num_evaluations <= params.max_iterations;
       ++num_evaluations) {
    // A value x falls outside the bracket [xₗ,xᵤ] whenever:
    //   (x-xₗ)⋅(x-xᵤ) > 0
    // Multiplying by f'² (which is positive):
    //   [(x-xₗ)f']⋅[f'(x-xᵤ)] > 0
    // Now, if x corresponds to the Newton update x = root - f / f', the above
    // inequality becomes:
    //   [(root-xₗ)f'-f]⋅[(root-xᵤ)f'-f)] > 0
    // which avoids the the division by f'.
    // const bool newton_falls_outside_bracket =
    //  ((root - x_upper) * df - f) * ((root - x_lower) * df - f) > 0.0;
    const double x = root - f / df;
    const bool newton_falls_outside_bracket = x < x_lower || x > x_upper;
    if (newton_falls_outside_bracket ||
        abs(2.0 * f) > abs(previous_minus_dx * df)) {
      // Bisection: Newton's method would either take us out of bounds or is not
      // reducing the size of the bracket fast enough.
      // Bisection updates root to:
      //   root = (x_upper + x_lower)/2
      // Given this update rule, whether the previous root was located at
      // x_lower or x_upper, the magnitude of the update is:
      //  dx = x - x_previous = (x_upper - x_lower)/2
      // With this definition of dx, the root can be written as:
      //  root = x_lower + dx
      previous_minus_dx = minus_dx;  // Save (minus) dx before updating it.
      minus_dx = 0.5 * (x_lower - x_upper);
      // Update root.
      // N.B. This way of updating the root will lead to root == x_lower if the
      // value of minus_dx is insignificant compared to x_lower when using
      // floating point precision. This fact is used in the termination check
      // below to exit whenever a user specifies abs_tolerance = 0.
      root = x_lower - minus_dx;
      // N.B. There is an extra space before "Bisect" so the console output
      // matches the indentation for the "Newton" case below.
      DRAKE_LOGGER_DEBUG(
          " Bisect. k = {:d}. x = {:10.4g}. [x_lower, x_upper] = [{:10.4g}, "
          "{:10.4g}]. dx = {:10.4g}. f = {:10.4g}. dfdx = {:10.4g}.",
          num_evaluations, root, x_lower, x_upper, -minus_dx, f, df);
      if (x_lower == root) {
        return std::make_pair(root, num_evaluations);
      }
    } else {
      // Newton method.
      previous_minus_dx = minus_dx;
      minus_dx = f / df;
      double previous_root = root;
      // N.B. This update will leave root unchanged if minus_dx is negligible
      // when compared to root using floating point precision. This fact is used
      // in the termination check below to exit whenever a user specifies
      // abs_tolerance = 0.
      root -= minus_dx;
      DRAKE_LOGGER_DEBUG(
          "Newton. k = {:d}. x = {:10.4g}. [x_lower, x_upper] = [{:10.4g}, "
          "{:10.4g}]. dx = {:10.4g}. f = {:10.4g}. dfdx = {:10.4g}.",
          num_evaluations, root, x_lower, x_upper, -minus_dx, f, df);
      if (previous_root == root) {
        // If previous_root equals root "after" the update, it means that
        // minus_dx is small compared to root's value, in floating point
        // precision.
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

    // Update the bracket around root to guarantee that there exist a root
    // within the interval [x_lower, x_upper].
    if (f * f_upper < 0.0) {
      x_lower = root;
      f_lower = f;
    } else {
      x_upper = root;
      f_upper = f;
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
