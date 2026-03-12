// Copyright (c) 2021, National Technology & Engineering Solutions of Sandia,
// LLC (NTESS). Copyright (c) 2022, Battelle Memorial Institute
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_AERO_SPECIES_HPP
#define MAM4XX_AERO_SPECIES_HPP

#include "mam4_types.hpp"

#include <limits>
#include <map>
#include <string>
#include <vector>

namespace mam4 {

/// @struct AeroSpecies
/// This type represents an aerosol species.
struct AeroSpecies final {
  // Molecular weight [kg/mol]
  const Real molecular_weight;

  /// Material density [kg/m^3]
  const Real density;

  /// Hygroscopicity
  const Real hygroscopicity;
};

} // namespace mam4

#endif
