// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_HPP
#define MAM4XX_HPP

// This header makes all MAM4 processes available.

#include <haero/aero_process.hpp>
#include <mam4xx/aero_config.hpp>
#include <mam4xx/aero_model.hpp>
#include <mam4xx/aging.hpp>
#include <mam4xx/calcsize.hpp>
#include <mam4xx/coagulation.hpp>
#include <mam4xx/convproc.hpp>
#include <mam4xx/drydep.hpp>
#include <mam4xx/gas_chem.hpp>
#include <mam4xx/gas_chem_mechanism.hpp>
#include <mam4xx/gasaerexch.hpp>
#include <mam4xx/hetfrz.hpp>
#include <mam4xx/lin_strat_chem.hpp>
#include <mam4xx/mam4_types.hpp>
#include <mam4xx/mo_chm_diags.hpp>
#include <mam4xx/mo_photo.hpp>
#include <mam4xx/mo_setext.hpp>
#include <mam4xx/mo_setsox.hpp>
#include <mam4xx/ndrop.hpp>
#include <mam4xx/nucleate_ice.hpp>
#include <mam4xx/nucleation.hpp>
#include <mam4xx/rename.hpp>
#include <mam4xx/water_uptake.hpp>
#include <mam4xx/wet_dep.hpp>

namespace mam4 {

using NucleationProcess = haero::AeroProcess<AeroConfig, Nucleation>;
using GasAerExchProcess = haero::AeroProcess<AeroConfig, GasAerExch>;
using CoagulationProcess = haero::AeroProcess<AeroConfig, Coagulation>;
using CalcSizeProcess = haero::AeroProcess<AeroConfig, CalcSize>;
using ConvProcProcess = haero::AeroProcess<AeroConfig, ConvProc>;
using AgingProcess = haero::AeroProcess<AeroConfig, Aging>;
using RenameProcess = haero::AeroProcess<AeroConfig, Rename>;
using HetfrzProcess = haero::AeroProcess<AeroConfig, Hetfrz>;
using NucleateIceProcess = haero::AeroProcess<AeroConfig, NucleateIce>;
using WetDepositionProcess = haero::AeroProcess<AeroConfig, WetDeposition>;
using DryDepProcess = haero::AeroProcess<AeroConfig, DryDep>;
using WaterUptakeProcess = haero::AeroProcess<AeroConfig, Water_Uptake>;

} // namespace mam4

#endif
