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

// the testing namespace contains functions that are useful only within tests,
// not to be used in production code
namespace mam4::testing {

constexpr int default_fpes = FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW;

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
