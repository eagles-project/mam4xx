#ifndef MAM4XX_HPP
#define MAM4XX_HPP

// This header makes all MAM4 processes available.

#include <haero/aero_process.hpp>
#include <mam4xx/aero_config.hpp>
#include <mam4xx/aging.hpp>
#include <mam4xx/calcsize.hpp>
#include <mam4xx/gasaerexch.hpp>
#include <mam4xx/mam4_types.hpp>
#include <mam4xx/nucleation.hpp>

namespace mam4 {

using NucleationProcess = haero::AeroProcess<AeroConfig, Nucleation>;
using GasAerExchProcess = haero::AeroProcess<AeroConfig, GasAerExch>;
using CalcSizeProcess = haero::AeroProcess<AeroConfig, CalcSize>;
using AgingProcess = haero::AeroProcess<AeroConfig, Aging>;

} // namespace mam4

#endif
