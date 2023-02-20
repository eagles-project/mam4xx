// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4_TYPES_HPP
#define MAM4_TYPES_HPP

// This header defines common Haero aliases for use in MAM4.

#include <haero/aero_species.hpp>
#include <haero/atmosphere.hpp>
#include <haero/constants.hpp>
#include <haero/floating_point.hpp>
#include <haero/gas_species.hpp>

namespace mam4 {

using DeviceType = haero::DeviceType;
using Constants = haero::Constants;
using Real = haero::Real;
using ColumnView = haero::ColumnView;
template <typename ST> using FloatingPoint = haero::FloatingPoint<ST>;
using Atmosphere = haero::Atmosphere;
using AeroSpecies = haero::AeroSpecies;
using GasSpecies = haero::GasSpecies;
using ThreadTeam = haero::ThreadTeam;

} // namespace mam4

#endif
