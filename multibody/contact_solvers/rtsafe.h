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

  // TODO: needed?
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
  double minus_dx = (x_lower - x_upper);
  auto [f, df] = function(root);

  auto do_bisection = [&x_upper, &x_lower]() {
    const double dx_negative = 0.5 * (x_lower - x_upper);
    // Update root.
    // N.B. This way of updating the root will lead to root == x_lower if
    // the value of minus_dx is insignificant compared to x_lower when using
    // floating point precision. This fact is used in the termination check
    // below to exit whenever a user specifies abs_tolerance = 0.
    const double x = x_lower - dx_negative;

    return std::make_pair(x, dx_negative);
  };

  auto do_newton = [&f, &df, &root]() {
    const double dx_negative = f / df;
    double x = root;
    x -= dx_negative;
    return std::make_pair(x, dx_negative);
  };

  for (int num_evaluations = 1; num_evaluations <= params.max_iterations;
       ++num_evaluations) {
    if (f == 0) return std::make_pair(root, num_evaluations);

    // N.B. Notice this check is always true for df = 0 (and f != 0 since we
    // ruled that case out above). Therefore Newton is only called when df != 0,
    // and the search direction is well defined.
    const bool newton_is_slow = 2.0 * abs(f) > abs(minus_dx * df);

    if (newton_is_slow) {
      std::tie(root, minus_dx) = do_bisection();
      DRAKE_LOGGER_DEBUG("Bisect. k = {:d}.", num_evaluations);
    } else {
      std::tie(root, minus_dx) = do_newton();
      PRINT_VAR(root);
      PRINT_VAR(minus_dx);
      const bool outside_bracket = root < x_lower || root > x_upper;
      if (outside_bracket) {
        std::tie(root, minus_dx) = do_bisection();
        DRAKE_LOGGER_DEBUG("Bisect. k = {:d}.", num_evaluations);
      } else {
        DRAKE_LOGGER_DEBUG("Newton. k = {:d}.", num_evaluations);
      }
    }

    DRAKE_LOGGER_DEBUG(
        "x = {:10.4g}. [x_lower, x_upper] = [{:10.4g}, "
        "{:10.4g}]. dx = {:10.4g}. f = {:10.4g}. dfdx = {:10.4g}.",
        root, x_lower, x_upper, -minus_dx, f, df);

    if (abs(minus_dx) < params.abs_tolerance)
      return std::make_pair(root, num_evaluations);

    // The one evaluation per iteration.
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
