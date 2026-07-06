// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_TESTING_HPP
#define MAM4XX_TESTING_HPP

#include <mam4xx/aero_config.hpp>
#include <mam4xx/atmosphere.hpp>
#include <mam4xx/surface.hpp>

#include <cfenv>
#include <cstring>

#define STRLCPY_ON_DEVICE(dst, src, n)                                         \
  {                                                                            \
    size_t src_len;                                                            \
    for (src_len = 0; src_len < n - 1 && src[src_len]; ++src_len)              \
      dst[src_len] = src[src_len];                                             \
    dst[src_len] = 0;                                                          \
  }

// use this macro in place of REQUIRE when testing predicates on device
#define REQUIRE_ON_DEVICE(results, p)                                          \
  if (!(p)) {                                                                  \
    for (int i = 0; i < mam4::testing::max_num_on_device_test_results; ++i) {  \
      if (!results(i).failed) {                                                \
        results(i).failed = true;                                              \
        STRLCPY_ON_DEVICE(results[i].predicate, #p, 1024);                     \
        STRLCPY_ON_DEVICE(results[i].file, __FILE__, 256);                     \
        results[i].line_number = __LINE__;                                     \
        break;                                                                 \
      }                                                                        \
    }                                                                          \
  }

// the testing namespace contains functions that are useful only within tests,
// not to be used in production code
namespace mam4::testing {

constexpr int default_fpes = FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW;

// we use a view of TestResult structs to convey on-device test failures to the
// host for reporting
struct OnDeviceTestResult {
  bool failed;
  char predicate[1024];
  char file[256];
  int line_number;
};
using OnDeviceTestResultView = mam4::DeviceType::view_1d<OnDeviceTestResult>;

constexpr int max_num_on_device_test_results = 128;

/// creates a view to store on-device test results -- pass this to any lambdas
/// dispatching tests to device
OnDeviceTestResultView create_on_device_test_results();

/// reports on-device test results after they have run, invoking an approprirate
/// Catch2 mechanism indicate any failures
void report_on_device_test_results(const OnDeviceTestResultView results);

/// creates an Atmosphere object that stores a column of data with the given
/// number of vertical levels and the given planetary boundary height
/// @param [in] num_levels the number of vertical levels per column stored by
///                        the state
/// @param [in] pblh The column-specific planetary boundary height [m],
///                  computed by the host model
Atmosphere create_atmosphere(int num_levels, Real pblh);

/// Creates a standalone ColumnView that uses resources allocated by a memory
/// pool.
ColumnView create_column_view(int num_levels);

// creates a Prognostics object with the given number of vertical levels and
// a set of newly-allocated views, managed using a testing column data pool
Prognostics create_prognostics(int num_levels);

// creates a Diagnostics object with the given number of vertical levels and
// a set of newly-allocated views, managed using a testing column data pool
Diagnostics create_diagnostics(int num_levels);

// creates a Tendencies object with the given number of vertical levels and
// a set of newly-allocated views, managed using a testing column data pool
Tendencies create_tendencies(int num_levels);

// creates a Surface object
Surface create_surface();

/// Call this at the end of a testing session to delete all ColumnViews
/// allocated by create_column_view. This is called by mam4xx's implementation
/// of ekat_finalize_test_session, which is called automatically at the end of
/// each Catch2-powered unit test.
void finalize();

} // namespace mam4::testing

#endif
