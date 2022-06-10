#pragma once

#include <functional>
#include <iostream>
#include <tuple>
#include <utility>

#include "drake/common/drake_copyable.h"
#include "drake/common/drake_throw.h"
#include "drake/common/text_logging.h"

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {

class Bracket {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(Bracket);

  Bracket(double x_lower, double f_lower, double x_upper, double f_upper)
      : x_lower_(x_lower),
        f_lower_(f_lower),
        x_upper_(x_upper),
        f_upper_(f_upper) {
    DRAKE_DEMAND(x_lower < x_upper);
    DRAKE_DEMAND(has_different_sign(f_lower, f_upper));
  }

  bool inside(double x) {
    return x_lower_ <= x && x <= x_upper_;
  }

  void Update(double x, double f) {
    if (has_different_sign(f, f_upper_)) {
      x_lower_ = x;
      f_lower_ = f;
    } else {
      x_upper_ = x;
      f_upper_ = f;
    }
  }
  
  double x_lower() const { return x_lower_; }
  double x_upper() const { return x_upper_; }
  double f_lower() const { return f_lower_; }
  double f_upper() const { return f_upper_; }

 private:
  static bool has_different_sign(double a, double b) {
    return std::signbit(a) ^ std::signbit(b);
  }
  double x_lower_;
  double f_lower_;
  double x_upper_;
  double f_upper_;
};

/*
  Uses a Newton-Raphson method to compute a root of `function` within the
  bracket [x_lower, x_upper]. This method stops when the difference between the
  previous iterate xᵏ and the next iteration xᵏ⁺¹ is below the absolute
  tolerance `abs_tolerance`, i.e. when |xᵏ⁺¹ - xᵏ| < abs_tolerance.

  This method iteratively shrinks the bracket containing the root. Moreover, it
  switches to bisection whenever a Newton iterate falls outside the bracket or
  when Newton's method is slow. Using this procedure, this method is guaranteed
  to find a root (which might not be unique) within [x_lower, x_upper], with
  accuracy given by `abs_tolerance`.

  This method expects that sign(function(x_lower)) != sign(function(x_upper)).
  For continuous functions, this ensures there exists a root in [x_lower,
  x_upper]. For discontinuous functions, the solver "sees" a discontinuity as a
  sharp transition within a narrow gap of size `abs_tolerance`. Therefore the
  solver will return a "root" located at where the discontinuity occurs. For
  instance, consider the function y(x) = 1/(x-c). While clearly discontinuous at
  x = c, this method will return c as the root whenever c is within the supplied
  bracket. Another example more common in practice is the function y(x) = x +
  H(x) - 1/2, with H(x) the Heaviside function. While discontinuous at x = 0,
  this method will return x = 0 as the root.

  @returns the pair (root, number_of_evaluations)

  @pre x_lower <= x_upper
  @pre x_guess is in [x_lower, x_upper]
  @pre sign(function(x_lower)) != sign(function_x_upper)
  @pre abs_tolerance > 0
  @pre max_iterations > 0
*/
std::pair<double, int> DoNewtonWithBisectionFallback(
    const std::function<std::pair<double, double>(double)>& function,
    Bracket bracket, double x_guess, double abs_tolerance,
    int max_iterations);

}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake
