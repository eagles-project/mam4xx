#ifndef MAM4XX_HPP
#define MAM4XX_HPP

// This header makes all MAM4 processes available.

#include "mam4_types.hpp"

#include <haero/aero_process.hpp>

#include "aero_config.hpp"
#include "gasaerexch.hpp"
#include "nucleation.hpp"

namespace mam4 {

using NucleationProcess = haero::AeroProcess<AeroConfig, Nucleation>;
using GasAerExchProcess = haero::AeroProcess<AeroConfig, GasAerExch>;

} // namespace mam4

#endif
