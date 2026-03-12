// Copyright (c) 2021, National Technology & Engineering Solutions of Sandia,
// LLC (NTESS). Copyright (c) 2022, Battelle Memorial Institute
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_ROOTFINDERS_HPP
#define MAM4XX_ROOTFINDERS_HPP

#include <mam4xx/floating_point.hpp>

namespace mam4 {
namespace math {

/** @brief Scalar rootfinding algorithm that employs Newton's method.
  This algorithm has a quadratic convergence rate but it is not guaranteed to
  converge. It may converge to an incorrect root if a poor initial guess is
  chosen. It requires both function values and derivative values.
*/
template <typename ScalarFunction> struct NewtonSolver {
  using value_type = typename ScalarFunction::value_type;

  /// maximum number of iterations allowed per solve
  static constexpr int max_iter = 200;
  /// solution
  value_type xroot;
  /// tolerance
  Real conv_tol;
  /// iteration counter
  int counter;
  /// absolute difference between two consecutive iterations
  value_type iter_diff;
  /// Scalar function whose root we need
  const ScalarFunction &f;
  /// true if a failure condition is met
  bool fail;

  /** @brief Constructor.

    @param [in] x0 Initial guess for the rootfinding algorithm
    @param [in] a0 Unused. Required to have unified interface for all solvers.
    @param [in] b0 Unused. Required to have unified interface for all solvers.
    @param [in] tol Convergence tolerance that determines when a root is found.
    @param [in] fn ScalarFunction instance whose root needs to be found
  */
  KOKKOS_INLINE_FUNCTION
  NewtonSolver(const value_type x0, const value_type a0, const value_type b0,
               const Real &tol, const ScalarFunction &fn)
      : xroot(x0), conv_tol(tol), counter(0), iter_diff(max()), f(fn),
        fail(false) {}

  /// Solves for the root.  Prints a warning message if the convergence
  /// tolerance is not met before the maximum number of iterations is achieved.
  KOKKOS_INLINE_FUNCTION
  value_type solve() { return solve_impl<value_type>(); }

  template <typename VT>
  KOKKOS_INLINE_FUNCTION
      typename std::enable_if<std::is_floating_point<VT>::value,
                              value_type>::type
      solve_impl() {
    bool keep_going = true;
    value_type xnp1;
    value_type f_deriv = 0;
    while (keep_going) {
      ++counter;
      f_deriv = f.derivative(xroot);
      if (FloatingPoint<value_type>::zero(f_deriv)) {
        xroot = nan("");
        keep_going = false;
        fail = true;
        break;
      }
      xnp1 = xroot - f(xroot) / f_deriv;
      iter_diff = abs(xnp1 - xroot);
      keep_going = !(FloatingPoint<value_type>::zero(iter_diff, conv_tol));
      EKAT_KERNEL_ASSERT_MSG(counter <= max_iter,
                             "NewtonSolver: max iterations");
      if (counter > max_iter) {
        keep_going = false;
        fail = true;
      }
      if (isnan(xnp1)) {
        keep_going = false;
        fail = true;
      }
      xroot = xnp1;
    }
    return xroot;
  }
};

/** padding factor stops BracketedNewtonSolver from getting stuck at an endpoint

  Larger values tend to require more iterations for the solver to converge.
*/
static constexpr Real bracket_pad_factor = 1e-7;

/** @brief Newton iterations protected by a bracket that prevents Newton from
going outside its bounds.

  This solver is a compromise between the speed of Newton's method and the
robustness of the bisection method. In addition to the requirement that the
initial interval contains a root, this solver requires that function value at
the initial interval endpoints have opposite sign.
*/
template <typename ScalarFunction> struct BracketedNewtonSolver {
  using value_type = typename ScalarFunction::value_type;
  static constexpr int max_iter = 200;
  value_type xroot;
  value_type a;
  value_type b;
  Real conv_tol;
  value_type fa;
  value_type fx;
  value_type fb;
  const ScalarFunction &f;
  int counter;
  value_type iter_diff;
  /// true if a failure condition is met
  bool fail;

  /** @brief Constructor.

    @param [in] x0 x0 Initial guess for the rootfinding algorithm
    @param [in] a0 Left endpoint of the initial interval
    @param [in] b0 Right endpoint of the initial interval
    @param [in] tol Convergence tolerance that determines when a root is found.
    @param [in] fn ScalarFunction instance whose root needs to be found
  */
  KOKKOS_INLINE_FUNCTION
  BracketedNewtonSolver(const value_type x0, const value_type a0,
                        const value_type b0, const Real tol,
                        const ScalarFunction &fn)
      : xroot(x0), a(a0), b(b0), conv_tol(tol), fa(fn(value_type(a0))),
        fx(fn(x0)), fb(fn(value_type(b0))), f(fn), counter(0), iter_diff(max()),
        fail(false) {
    EKAT_KERNEL_ASSERT(b - a > 0.0);
    EKAT_KERNEL_ASSERT(fa * fb < 0.0);
  }

  KOKKOS_INLINE_FUNCTION
  value_type solve() { return solve_impl<value_type>(); }

  template <typename VT>
  KOKKOS_INLINE_FUNCTION
      typename std::enable_if<std::is_floating_point<VT>::value,
                              value_type>::type
      solve_impl() {
    bool keep_going = true;
    while (keep_going) {
      ++counter;
      // newton step
      value_type x = xroot - fx / f.derivative(xroot);
      // safeguard: require x to be inside current bracket
      // assure progress: guard against tiny steps
      const Real pad_fac = bracket_pad_factor;
      const Real pad = pad_fac * (b - a);
      if (!FloatingPoint<Real>::in_bounds(x, a + pad, b - pad)) {
        x = 0.5 * (a + b);
      }
      fx = f(x);
      // update bracket
      if (fx * fa > 0) {
        a = x;
        fa = fx;
      } else {
        b = x;
        fb = fx;
      }
      // check convergence
      iter_diff = abs(x - xroot);
      keep_going = !FloatingPoint<value_type>::zero(iter_diff, conv_tol);
      // prevent infinite loops
      EKAT_KERNEL_ASSERT_MSG(counter <= max_iter,
                             "BracketedNewtonSolver: max iterations");
      if (counter > max_iter) {
        keep_going = false;
        fail = true;
      }
      if (isnan(x)) {
        keep_going = false;
        fail = true;
      }
      xroot = x;
    }
    return xroot;
  }
};

/** @brief Scalar rootfinding algorithm that employs the recursive bisection
  method. This method has only a linear convergence rate, but it is guaranteed
  to converge if the initial interval contains a root.

  The Legendre polynomials above demonstrate the required ScalarFunction
  interface.

  This solver requires the initial interval to contain a root.

  For an application example, see KohlerPolynomial.

*/
template <typename ScalarFunction> struct BisectionSolver {
  using value_type = typename ScalarFunction::value_type;

  /// maximum number of iterations allowed
  static constexpr int max_iter = 200;
  /// solution
  value_type xroot;
  /// left endpoint of root search interval
  value_type a;
  /// right endpoint of root search interval
  value_type b;
  /// tolerance
  Real conv_tol;
  /// function value at left endpoint of search interval
  value_type fa;
  /// next iteration solution
  value_type xnp1;
  /// iteration counter
  int counter;
  /// width of the current search interval
  value_type iter_diff;
  /// scalar function whose root we need
  const ScalarFunction &f;
  /// true if a failure condition is met
  bool fail;

  /** @brief Constructor.

    @param [in] x0 Unused. Required to have unified interface for all solvers.
    @param [in] a0 Left endpoint of the initial interval
    @param [in] b0 Right endpoint of the initial interval
    @param [in] tol Convergence tolerance that determines when a root is found.
    @param [in] fn ScalarFunction instance whose root needs to be found
  */
  KOKKOS_INLINE_FUNCTION
  BisectionSolver(const value_type x0, const value_type a0, const value_type b0,
                  const Real tol, const ScalarFunction &fn)
      : xroot(0.5 * (a0 + b0)), a(a0), b(b0), conv_tol(tol), fa(fn(a0)),
        xnp1(0.5 * (a0 + b0)), counter(0), iter_diff(b0 - a0), f(fn),
        fail(false) {}

  /// Solves for the root.  Prints a warning message if the convergence
  /// tolerance is not met before the maximum number of iterations is achieved.
  KOKKOS_INLINE_FUNCTION
  value_type solve() { return solve_impl<value_type>(); }

  template <typename VT>
  KOKKOS_INLINE_FUNCTION
      typename std::enable_if<std::is_floating_point<VT>::value,
                              value_type>::type
      solve_impl() {
    bool keep_going = true;
    while (keep_going) {
      ++counter;
      const value_type fx = f(xroot);
      xnp1 = 0.5 * (a + b);
      if (fx * fa < 0) {
        b = xroot;
      } else {
        a = xroot;
        fa = fx;
      }
      iter_diff = b - a;
      xroot = xnp1;
      keep_going = !(FloatingPoint<value_type>::zero(iter_diff, conv_tol));
      EKAT_KERNEL_ASSERT_MSG(counter <= max_iter,
                             "BisectionSolver: max iterations");
      if (counter > max_iter) {
        keep_going = false;
        fail = true;
      }
      if (isnan(xnp1)) {
        keep_going = false;
        fail = true;
      }
    }
    return xroot;
  }
};

} // namespace math
} // namespace mam4
#endif
