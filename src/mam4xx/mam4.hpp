#ifndef MAM4XX_HPP
#define MAM4XX_HPP

// This header makes all MAM4 processes available.

#include <mam4xx/aero_config.hpp>
#include <mam4xx/gasaerexch.hpp>
#include <mam4xx/mam4_types.hpp>
#include <mam4xx/nucleation.hpp>

#include <haero/aero_process.hpp>

namespace mam4 {

using NucleationProcess = haero::AeroProcess<AeroConfig, Nucleation>;
using GasAerExchProcess = haero::AeroProcess<AeroConfig, GasAerExch>;

} // namespace mam4

#endif
