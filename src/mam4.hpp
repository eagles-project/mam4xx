#ifndef MAM4XX_HPP
#define MAM4XX_HPP

// This header makes available all MAM4 processes.

#include "nucleation_impl.hpp"
#include "gasaerexch_impl.hpp"

#include <haero/aero_process.hpp>

namespace mam4 {

using NucleationProcess = AeroProcess<AeroConfig, NucleationImpl>;
using GasAerExchProcess = AeroProcess<AeroConfig, GasAerExchImpl>;

} // namespace mam4

#endif
