// Copyright (c) 2021, National Technology & Engineering Solutions of Sandia,
// LLC (NTESS). Copyright (c) 2022, Battelle Memorial Institute
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_GAS_SPECIES_HPP
#define MAM4XX_GAS_SPECIES_HPP

#include <mam4xx/mam4_config.hpp>

namespace mam4 {

/// @struct GasSpecies
/// This type represents a gas that participates in one or more aerosol
/// microphysics parameterizations.
struct GasSpecies final {
  /// Molecular weight [kg/mol]
  Real molecular_weight;
};

} // namespace mam4
#endif
