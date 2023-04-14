// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_TESTING_HPP
#define MAM4XX_TESTING_HPP

#include <mam4xx/aero_config.hpp>

#include <haero/testing.hpp>

// the testing namespace contains functions that are useful only within tests,
// not to be used in production code
namespace mam4::testing {

// we forward testing functions from Haero
using namespace haero::testing;

// creates a Prognostics object with the given number of vertical levels and
// a set of newly-allocated views, managed using Haero's testing column data
// pool
Prognostics create_prognostics(int num_levels);

// creates a Diagnostics object with the given number of vertical levels and
// a set of newly-allocated views, managed using Haero's testing column data
// pool
Diagnostics create_diagnostics(int num_levels);

// creates a Tendencies object with the given number of vertical levels and
// a set of newly-allocated views, managed using Haero's testing column data
// pool
Tendencies create_tendencies(int num_levels);

} // namespace mam4::testing

#endif
