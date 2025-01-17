#pragma once

#include <cmath>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "drake/common/drake_assert.h"
#include "drake/common/drake_copyable.h"
#include "drake/common/drake_deprecated.h"
#include "drake/common/polynomial.h"

namespace drake {

/** A scalar multi-variate polynomial containing sines and cosines.
 *
 * TrigPoly represents a Polynomial some of whose variables actually represent
 * the sines or cosines of other variables.  Sines and cosines of first-order
 * polynomials (affine combinations of variables) are decomposed into
 * polynomials of the sines and cosines of individual variables via the
 * Prosthaphaeresis formulae.
 *
 * Any variables which will appear in the arguments to trigonometric functions
 * must be declared in the "SinCosMap" (created automatically by most TrigPoly
 * constructors); attempting to, e.g., use sin(x) without first creating a
 * SinCosMap mapping for 'x' will result in an exception.
 *
 * The same variable may not appear more than once in the sin_cos_map,
 * regardless of position.
 *
 * For example:
 * \code
 * Polynomial base_x("x"), s("s"), c("c");
 * TrigPoly x(base_x, s, c)  // This "x" knows that s = sin(x)
 *                           // and that c = cos(x)
 * cout << sin(x)                     // emits "s1"
 * cout << sin(x) * x                 // emits "x1*s1"
 * cout << sin(x + x) * x             // emits "x1*s1*c1 + x1*c1*s1"
 * \endcode
 *
 * NOTE:  Certain analyses may not succeed when individual Monomials contain
 * both x and sin(x) or cos(x) terms.  This restriction is not currently
 * enforced programmatically.
 * <!-- TODO(ggould-tri): Fix this in the future. -->
 */
template <typename T = double>
class DRAKE_DEPRECATED(
    "2022-07-01",
    "TrigPoly is deprecated. Please use symbolic::Expression and the "
    "SinCosSubstitution instead.") TrigPoly final {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(TrigPoly)

  typedef Polynomial<T> PolyType;
  typedef typename PolyType::VarType VarType;
  struct SinCosVars {
    VarType s;
    VarType c;

    bool operator==(const struct SinCosVars& other) const {
      return (s == other.s) && (c == other.c);
    }
  };
  typedef std::map<VarType, SinCosVars> SinCosMap;

  template <typename Rhs, typename Lhs>
  struct Product {
    typedef decltype(static_cast<Rhs>(0) * static_cast<Lhs>(0)) type;
  };

  /// Constructs a vacuous TrigPoly.
  TrigPoly() {}

  /// Constructs a constant TrigPoly.
  // NOLINTNEXTLINE(runtime/explicit) This conversion is desirable.
  TrigPoly(const T& scalar) : poly_(scalar) {}

  /**
   * Constructs a TrigPoly on the associated Polynomial p with no associated
   * trigonometric correspondences.
   */
  explicit TrigPoly(const PolyType& p) : poly_(p) {}

  /**
   * Constructs a TrigPoly on the associated Polynomial p, but with the
   * additional information about sin and cos relations in _sin_cos_map.
   */
  TrigPoly(const PolyType& p, const SinCosMap& _sin_cos_map)
      : poly_(p), sin_cos_map_(_sin_cos_map) {
    // The provided _sin_cos_map might have extraneous entries; clip them.
    std::set<VarType> vars_in_use = p.GetVariables();
    std::set<VarType> vars_seen_in_map;
    for (const auto& sin_cos_entry : _sin_cos_map) {
      if (!(vars_in_use.count(sin_cos_entry.first) ||
            vars_in_use.count(sin_cos_entry.second.c) ||
            vars_in_use.count(sin_cos_entry.second.s))) {
        sin_cos_map_.erase(sin_cos_entry.first);
      }
      DRAKE_ASSERT(!vars_seen_in_map.count(sin_cos_entry.first));
      DRAKE_ASSERT(!vars_seen_in_map.count(sin_cos_entry.second.s));
      DRAKE_ASSERT(!vars_seen_in_map.count(sin_cos_entry.second.c));
      vars_seen_in_map.insert(sin_cos_entry.first);
      vars_seen_in_map.insert(sin_cos_entry.second.s);
      vars_seen_in_map.insert(sin_cos_entry.second.c);
    }
  }

  /**
   * Constructs a TrigPoly version of q, but with the additional information
   * that the variables s and c represent the sine and cosine of q.
   */
  TrigPoly(const PolyType& q, const PolyType& s, const PolyType& c) {
    if ((q.GetDegree() != 1) || (s.GetDegree() != 1) || (c.GetDegree() != 1))
      throw std::runtime_error(
          "q, s, and c must all be simple polynomials (in the msspoly sense)");

    poly_ = q;
    SinCosVars sc;
    sc.s = s.GetSimpleVariable();
    sc.c = c.GetSimpleVariable();
    sin_cos_map_[q.GetSimpleVariable()] = sc;
  }

  ~TrigPoly() = default;

  /// Returns the underlying Polynomial for this TrigPoly.
  const PolyType& poly(void) const { return poly_; }

  /// Returns the SinCosMap for this TrigPoly.
  const SinCosMap& sin_cos_map(void) const { return sin_cos_map_; }

  /** A version of sin that handles TrigPoly arguments through ADL.
   *
   * Implements sin(x) for a TrigPoly x.
   *
   * x must be of degree 0 or 1, and must contain only variables that have
   * entries in its SinCosMap.
   */
  friend TrigPoly sin(const TrigPoly& p) {
    using std::abs;
    using std::copysign;
    using std::floor;

    const std::vector<typename PolyType::Monomial>& m = p.poly_.GetMonomials();
    if (m.empty()) {
      return TrigPoly(Polynomial<T>(sin(T(0))));
    } else if (p.poly_.GetDegree() > 1) {
      throw std::runtime_error(
          "sin of polynomials with degree > 1 is not supported");
    }

    // Base case of recursion is one mononomial with coefficient of 1.
    if (m.size() == 1 && abs(m[0].coefficient) == T(1)) {
      TrigPoly ret = p;
      if (m[0].terms.empty()) {  // then it's a constant
        ret.poly_ = Polynomial<T>(sin(m[0].coefficient));
      } else {
        typename SinCosMap::iterator iter =
            ret.sin_cos_map_.find(m[0].terms[0].var);
        if (iter == ret.sin_cos_map_.end())
          throw std::runtime_error(
              "tried taking the sin of a variable that does not exist in my "
              "sin_cos_map");
        ret.poly_.Subs(m[0].terms[0].var, iter->second.s);
      }
      return ret;
    }

    Polynomial<T> pa;
    Polynomial<T> pb(m.begin() + 1, m.end());
    if (abs(m[0].coefficient) == T(1)) {
      pa = Polynomial<T>(m[0].coefficient, m[0].terms);
    } else if (m[0].coefficient - floor(m[0].coefficient) ==
               T(0)) {
      // If the coefficient has integer magnitude greater than 1, recurse by
      // expanding out a term with coefficient of magnitude 1.
      auto unit = copysign(T(1), m[0].coefficient);
      pa = Polynomial<T>(unit, m[0].terms);
      pb += Polynomial<T>(m[0].coefficient - unit, m[0].terms);
    } else {
      throw std::runtime_error("Fractional coefficients not supported");
    }

    // Now handle the multi-monomial case recursively
    // sin(a+b+...) = sin(a)cos(b+...) + cos(a)sin(b+...)
    TrigPoly a(pa, p.sin_cos_map_);
    TrigPoly b(pb, p.sin_cos_map_);
    return sin(a) * cos(b) + cos(a) * sin(b);
  }

  /// A version of cos that handles TrigPoly arguments through ADL.
  /**
   * Implements cos(x) for a TrigPoly x.
   *
   * x must be of degree 0 or 1, and must contain only variables that have
   * entries in its SinCosMap.
   */
  friend TrigPoly cos(const TrigPoly& p) {
    using std::abs;
    using std::copysign;
    using std::floor;

    const std::vector<typename PolyType::Monomial>& m = p.poly_.GetMonomials();
    if (m.empty()) {
      return TrigPoly(Polynomial<T>(cos(T(0))));
    } else if (p.poly_.GetDegree() > 1) {
      throw std::runtime_error(
          "cos of polynomials with degree > 1 is not supported");
    }

    // Base case of recursion is one mononomial with coefficient of 1.
    if (m.size() == 1 && abs(m[0].coefficient) == T(1)) {
      TrigPoly ret = p;
      if (m[0].terms.empty()) {  // then it's a constant
        ret.poly_ = Polynomial<T>(cos(m[0].coefficient));
      } else {
        typename SinCosMap::iterator iter =
            ret.sin_cos_map_.find(m[0].terms[0].var);
        if (iter == ret.sin_cos_map_.end())
          throw std::runtime_error(
              "tried taking the sin of a variable that does not exist in my "
              "sin_cos_map");
        ret.poly_.Subs(m[0].terms[0].var, iter->second.c);
        if (m[0].coefficient == T(-1)) {
          ret *= -1;
        }  // cos(-q) => cos(q) => c (instead of -c)
      }
      return ret;
    }

    Polynomial<T> pa;
    Polynomial<T> pb(m.begin() + 1, m.end());
    if (abs(m[0].coefficient) == T(1)) {
      pa = Polynomial<T>(m[0].coefficient, m[0].terms);
    } else if (m[0].coefficient - floor(m[0].coefficient) ==
               T(0)) {
      // If the coefficient has integer magnitude greater than 1, recurse by
      // expanding out a term with coefficient of magnitude 1.
      auto unit = copysign(T(1), m[0].coefficient);
      pa = Polynomial<T>(unit, m[0].terms);
      pb += Polynomial<T>(m[0].coefficient - unit, m[0].terms);
    } else {
      throw std::runtime_error("Fractional coefficients not supported");
    }

    // Now handle the multi-monomial case recursively
    // cos(a+b+...) = cos(a)cos(b+...) - sin(a)sin(b+...)
    TrigPoly a(pa, p.sin_cos_map_);
    TrigPoly b(pb, p.sin_cos_map_);
    return cos(a) * cos(b) - sin(a) * sin(b);
  }

  /// Returns all of the base (non-sin/cos) variables in this TrigPoly.
  std::set<VarType> GetVariables() const {
    std::set<VarType> vars = poly_.GetVariables();
    for (const auto& sin_cos_item : sin_cos_map_) {
      vars.insert(sin_cos_item.first);
      vars.erase(sin_cos_item.second.s);
      vars.erase(sin_cos_item.second.c);
    }
    return vars;
  }

  /** Given a value for every variable in this expression, evaluates it.
   *
   * By analogy with Polynomial::EvaluateMultivariate().  Values must be
   * supplied for all base variables; supplying values for sin/cos variables
   * is an error.
   */
  template <typename U>
  typename Product<T, U>::type EvaluateMultivariate(
      const std::map<VarType, U>& var_values) const {
    std::map<VarType, U> all_var_values = var_values;
    for (const auto& sin_cos_item : sin_cos_map_) {
      DRAKE_ASSERT(!var_values.count(sin_cos_item.second.s));
      DRAKE_ASSERT(!var_values.count(sin_cos_item.second.c));
      all_var_values[sin_cos_item.second.s] =
          std::sin(var_values.at(sin_cos_item.first));
      all_var_values[sin_cos_item.second.c] =
          std::cos(var_values.at(sin_cos_item.first));
    }
    return poly_.EvaluateMultivariate(all_var_values);
  }

  /** Partially evaluates this expression, returning the resulting expression.
   *
   * By analogy with Polynomial::evaluatePartial.  Values must be supplied for
   * all base variables only; supplying values for sin/cos variables is an
   * error.
   */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  virtual TrigPoly<T> EvaluatePartial(
      const std::map<VarType, T>& var_values) const {
    std::map<VarType, T> var_values_with_sincos = var_values;
    for (const auto& sin_cos_item : sin_cos_map_) {
      DRAKE_ASSERT(!var_values.count(sin_cos_item.second.s));
      DRAKE_ASSERT(!var_values.count(sin_cos_item.second.c));
      if (!var_values.count(sin_cos_item.first)) {
        continue;
      }
      var_values_with_sincos[sin_cos_item.second.s] =
          std::sin(var_values.at(sin_cos_item.first));
      var_values_with_sincos[sin_cos_item.second.c] =
          std::cos(var_values.at(sin_cos_item.first));
    }
    return TrigPoly(poly_.EvaluatePartial(var_values_with_sincos),
                    sin_cos_map_);
  }
#pragma GCC diagnostic pop

  /// Compares two TrigPolys for equality.
  /**
   * Note that the question of equality of TrigPolys is a bit subtle.  It is
   * not immediately clear if two TrigPolys whose poly and sin_cos_map members
   * differ equivalently (eg, a + b (b = cos(a)) and a + c (c = cos(a))) should
   * be considered equal.
   *
   * For simplicity we only consider exactly equality rather than semantic
   * equivalence.  However that decision could reasonably revisited in the
   * future.
   */
  bool operator==(const TrigPoly& other) const {
    return (poly_ == other.poly_) && (sin_cos_map_ == other.sin_cos_map_);
  }

  TrigPoly& operator+=(const TrigPoly& other) {
    poly_ += other.poly_;
    sin_cos_map_.insert(other.sin_cos_map_.begin(), other.sin_cos_map_.end());
    return *this;
  }

  TrigPoly& operator-=(const TrigPoly& other) {
    poly_ -= other.poly_;
    sin_cos_map_.insert(other.sin_cos_map_.begin(), other.sin_cos_map_.end());
    return *this;
  }

  TrigPoly& operator*=(const TrigPoly& other) {
    poly_ *= other.poly_;
    sin_cos_map_.insert(other.sin_cos_map_.begin(), other.sin_cos_map_.end());
    return *this;
  }

  TrigPoly& operator+=(const T& scalar) {
    poly_ += scalar;
    return *this;
  }

  TrigPoly& operator-=(const T& scalar) {
    poly_ -= scalar;
    return *this;
  }

  TrigPoly& operator*=(const T& scalar) {
    poly_ *= scalar;
    return *this;
  }

  TrigPoly& operator/=(const T& scalar) {
    poly_ /= scalar;
    return *this;
  }

  const TrigPoly operator+(const TrigPoly& other) const {
    TrigPoly ret = *this;
    ret += other;
    return ret;
  }

  const TrigPoly operator-(const TrigPoly& other) const {
    TrigPoly ret = *this;
    ret -= other;
    return ret;
  }

  const TrigPoly operator-() const {
    TrigPoly ret = (*this) * T(-1);
    return ret;
  }

  const TrigPoly operator*(const TrigPoly& other) const {
    TrigPoly ret = *this;
    ret *= other;
    return ret;
  }

  friend const TrigPoly operator+(const TrigPoly& p,
                                  const T& scalar) {
    TrigPoly ret = p;
    ret += scalar;
    return ret;
  }

  friend const TrigPoly operator+(const T& scalar,
                                  const TrigPoly& p) {
    TrigPoly ret = p;
    ret += scalar;
    return ret;
  }

  friend const TrigPoly operator-(const TrigPoly& p,
                                  const T& scalar) {
    TrigPoly ret = p;
    ret -= scalar;
    return ret;
  }

  friend const TrigPoly operator-(const T& scalar,
                                  const TrigPoly& p) {
    TrigPoly ret = -p;
    ret += scalar;
    return ret;
  }

  friend const TrigPoly operator*(const TrigPoly& p,
                                  const T& scalar) {
    TrigPoly ret = p;
    ret *= scalar;
    return ret;
  }
  friend const TrigPoly operator*(const T& scalar,
                                  const TrigPoly& p) {
    TrigPoly ret = p;
    ret *= scalar;
    return ret;
  }

  const TrigPoly operator/(const T& scalar) const {
    TrigPoly ret = *this;
    ret /= scalar;
    return ret;
  }

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  friend std::ostream& operator<<(std::ostream& os,
                                  const TrigPoly<T>& tp) {
    os << tp.poly_;
    if (tp.sin_cos_map_.size()) {
      os << " where ";
      for (const auto& k_v_pair : tp.sin_cos_map_) {
        std::string var = PolyType::IdToVariableName(k_v_pair.first);
        std::string sin = PolyType::IdToVariableName(k_v_pair.second.s);
        std::string cos = PolyType::IdToVariableName(k_v_pair.second.c);
        os << sin << "=sin(" << var << "), "
           << cos << "=cos(" << var << "), ";
      }
    }
    return os;
  }
#pragma GCC diagnostic pop

 private:
  PolyType poly_;
  SinCosMap sin_cos_map_;
};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
template <typename T, int Rows, int Cols>
std::ostream& operator<<(
    std::ostream& os,
    const Eigen::Matrix<TrigPoly<T>, Rows, Cols>& tp_mat) {
  Eigen::Matrix<Polynomial<T>, Rows, Cols> poly_mat(
      tp_mat.rows(), tp_mat.cols());
  for (int i = 0; i < poly_mat.size(); i++) {
    poly_mat(i) = tp_mat(i).poly();
  }
  os << poly_mat;
  return os;
}

typedef TrigPoly<double> TrigPolyd;

/// A column vector of TrigPoly; used in several optimization classes.
typedef Eigen::Matrix<TrigPolyd, Eigen::Dynamic, 1> VectorXTrigPoly;

#pragma GCC diagnostic pop

}  // namespace drake
