// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_UTILS_HPP
#define MAM4XX_UTILS_HPP

#include <haero/math.hpp>
#include <mam4xx/conversions.hpp>
#include <mam4xx/gas_chem_mechanism.hpp>
// This file contains utility-type functions that are available for use by
// various processes, tests, etc.

namespace mam4::utils {

using Real = haero::Real;
using haero::max;
using haero::min;

// this function considers 'num' and returns either 'num' (already in bounds) or
// 'high'/'low' if num is outside the bounds
KOKKOS_INLINE_FUNCTION
Real min_max_bound(const Real &low, const Real &high, const Real &num) {
  return max(low, min(high, num));
}

// number of constituents in gas chemistry "work arrays"
KOKKOS_INLINE_FUNCTION
constexpr int gas_pcnst() {
  constexpr int gas_pcnst_ = mam4::gas_chemistry::gas_pcnst;
  return gas_pcnst_;
}

// Because CUDA C++ doesn't allow us to declare and use constants outside of
// KOKKOS_INLINE_FUNCTIONS, we define this macro that allows us to (re)define
// these constants where needed within two such functions so we don't define
// them inconsistently. Yes, it's the 21st century and we're still struggling
// with these basic things.
#define DECLARE_PROG_TRANSFER_CONSTANTS                                        \
  /* mapping of constituent indices to aerosol modes */                        \
  const auto Accum = mam4::ModeIndex::Accumulation;                            \
  const auto Aitken = mam4::ModeIndex::Aitken;                                 \
  const auto Coarse = mam4::ModeIndex::Coarse;                                 \
  const auto PC = mam4::ModeIndex::PrimaryCarbon;                              \
  const auto NoMode = mam4::ModeIndex::None;                                   \
  static const mam4::ModeIndex mode_for_cnst[gas_pcnst()] = {                  \
      NoMode, NoMode, NoMode, NoMode, NoMode, NoMode, /* gases (not aerosols)  \
                                                       */                      \
      Accum,  Accum,  Accum,  Accum,  Accum,  Accum,  Accum,                   \
      Accum,                                  /* 7 aero species + NMR */       \
      Aitken, Aitken, Aitken, Aitken, Aitken, /* 4 aero species + NMR */       \
      Coarse, Coarse, Coarse, Coarse, Coarse, Coarse, Coarse,                  \
      Coarse,                     /* 7 aero species + NMR */                   \
      PC,     PC,     PC,     PC, /* 3 aero species + NMR */                   \
  };                                                                           \
  /* mapping of constituent indices to aerosol species */                      \
  const auto SOA = mam4::AeroId::SOA;                                          \
  const auto SO4 = mam4::AeroId::SO4;                                          \
  const auto POM = mam4::AeroId::POM;                                          \
  const auto BC = mam4::AeroId::BC;                                            \
  const auto NaCl = mam4::AeroId::NaCl;                                        \
  const auto DST = mam4::AeroId::DST;                                          \
  const auto MOM = mam4::AeroId::MOM;                                          \
  const auto NoAero = mam4::AeroId::None;                                      \
  static const mam4::AeroId aero_for_cnst[gas_pcnst()] = {                     \
      NoAero, NoAero, NoAero, NoAero, NoAero, NoAero, /* gases (not aerosols)  \
                                                       */                      \
      SO4,    POM,    SOA,    BC,     DST,    NaCl,   MOM,                     \
      NoAero,                                 /* accumulation mode */          \
      SO4,    SOA,    NaCl,   MOM,    NoAero, /* aitken mode */                \
      DST,    NaCl,   SO4,    BC,     POM,    SOA,    MOM,                     \
      NoAero,                         /* coarse mode */                        \
      POM,    BC,     MOM,    NoAero, /* primary carbon mode */                \
  };                                                                           \
  /* mapping of constituent indices to gases */                                \
  const auto O3 = mam4::GasId::O3;                                             \
  const auto H2O2 = mam4::GasId::H2O2;                                         \
  const auto H2SO4 = mam4::GasId::H2SO4;                                       \
  const auto SO2 = mam4::GasId::SO2;                                           \
  const auto DMS = mam4::GasId::DMS;                                           \
  const auto SOAG = mam4::GasId::SOAG;                                         \
  const auto NoGas = mam4::GasId::None;                                        \
  static const mam4::GasId gas_for_cnst[gas_pcnst()] = {                       \
      O3,    H2O2,  H2SO4, SO2,   DMS,   SOAG,  NoGas, NoGas,                  \
      NoGas, NoGas, NoGas, NoGas, NoGas, NoGas, NoGas, NoGas,                  \
      NoGas, NoGas, NoGas, NoGas, NoGas, NoGas, NoGas, NoGas,                  \
      NoGas, NoGas, NoGas, NoGas, NoGas, NoGas, NoGas,                         \
  };

// Given a Prognostics object, transfers data for interstitial aerosols to the
// chemistry work array q, and cloudborne aerosols to the chemistry work array
// qqcw, both at vertical level k. The input and output quantities are stored as
// number/mass mixing ratios.
// NOTE: this mapping is chemistry-mechanism-specific (see mo_sim_dat.F90
// NOTE: in the relevant preprocessed chemical mechanism)
// NOTE: see mam4xx/aero_modes.hpp to interpret these mode/aerosol/gas
// NOTE: indices
KOKKOS_INLINE_FUNCTION
void transfer_prognostics_to_work_arrays(const mam4::Prognostics &progs,
                                         const int k, Real q[gas_pcnst()],
                                         Real qqcw[gas_pcnst()]) {
  DECLARE_PROG_TRANSFER_CONSTANTS

  // copy number/mass mixing ratios from progs to q and qqcw at level k,
  // converting them to VMR
  for (int i = 0; i < gas_pcnst(); ++i) {
    auto mode_index = mode_for_cnst[i];
    auto aero_id = aero_for_cnst[i];
    auto gas_id = gas_for_cnst[i];
    if (gas_id != NoGas) { // constituent is a gas
      int g = static_cast<int>(gas_id);
      q[i] = progs.q_gas[g](k);
      qqcw[i] = progs.q_gas[g](k);
    } else {
      int m = static_cast<int>(mode_index);
      if (aero_id != NoAero) { // constituent is an aerosol species
        int a = aerosol_index_for_mode(mode_index, aero_id);
        q[i] = progs.q_aero_i[m][a](k);
        qqcw[i] = progs.q_aero_c[m][a](k);
      } else { // constituent is a modal number mixing ratio
        int m = static_cast<int>(mode_index);
        q[i] = progs.n_mode_i[m](k);
        qqcw[i] = progs.n_mode_c[m](k);
      }
    }
  }
}

// Given work arrays with interstitial and cloudborne aerosol data, transfers
// them to the given Prognostics object at the kth vertical level. This is the
// "inverse operator" for transfer_prognostics_to_work_arrays, above.
KOKKOS_INLINE_FUNCTION
void transfer_work_arrays_to_prognostics(const Real q[gas_pcnst()],
                                         const Real qqcw[gas_pcnst()],
                                         mam4::Prognostics &progs,
                                         const int k) {
  DECLARE_PROG_TRANSFER_CONSTANTS

  // copy number/mass mixing ratios from progs to q and qqcw at level k,
  // converting them to VMR
  for (int i = 0; i < gas_pcnst(); ++i) {
    auto mode_index = mode_for_cnst[i];
    auto aero_id = aero_for_cnst[i];
    auto gas_id = gas_for_cnst[i];
    if (gas_id != NoGas) { // constituent is a gas
      int g = static_cast<int>(gas_id);
      progs.q_gas[g](k) = q[i];
    } else {
      int m = static_cast<int>(mode_index);
      if (aero_id != NoAero) { // constituent is an aerosol species
        int a = aerosol_index_for_mode(mode_index, aero_id);
        progs.q_aero_i[m][a](k) = q[i];
        progs.q_aero_c[m][a](k) = qqcw[i];
      } else { // constituent is a modal number mixing ratio
        int m = static_cast<int>(mode_index);
        progs.n_mode_i[m](k) = q[i];
        progs.n_mode_c[m](k) = qqcw[i];
      }
    }
  }
}

// converts the quantities in the work arrays q and qqcw from mass/number
// mixing ratios to volume/number mixing ratios
KOKKOS_INLINE_FUNCTION
void convert_work_arrays_to_vmr(const Real q[gas_pcnst()],
                                const Real qqcw[gas_pcnst()],
                                Real vmr[gas_pcnst()],
                                Real vmrcw[gas_pcnst()]) {
  DECLARE_PROG_TRANSFER_CONSTANTS

  for (int i = 0; i < gas_pcnst(); ++i) {
    auto mode_index = mode_for_cnst[i];
    auto aero_id = aero_for_cnst[i];
    auto gas_id = gas_for_cnst[i];
    if (gas_id != NoGas) { // constituent is a gas
      int g = static_cast<int>(gas_id);
      const Real mw = mam4::gas_species(g).molecular_weight;
      vmr[i] = mam4::conversions::vmr_from_mmr(q[i], mw);
      vmrcw[i] = mam4::conversions::vmr_from_mmr(qqcw[i], mw);
    } else {
      if (aero_id != NoAero) { // constituent is an aerosol species
        int a = aerosol_index_for_mode(mode_index, aero_id);
        const Real mw = mam4::aero_species(a).molecular_weight;
        vmr[i] = mam4::conversions::vmr_from_mmr(q[i], mw);
        vmrcw[i] = mam4::conversions::vmr_from_mmr(qqcw[i], mw);
      } else { // constituent is a modal number mixing ratio
        vmr[i] = q[i];
        vmrcw[i] = qqcw[i];
      }
    }
  }
}

// converts the quantities in the work arrays vmrs and vmrscw from mass/number
// mixing ratios to volume/number mixing ratios
KOKKOS_INLINE_FUNCTION
void convert_work_arrays_to_mmr(const Real vmr[gas_pcnst()],
                                const Real vmrcw[gas_pcnst()],
                                Real q[gas_pcnst()], Real qqcw[gas_pcnst()]) {
  DECLARE_PROG_TRANSFER_CONSTANTS

  for (int i = 0; i < gas_pcnst(); ++i) {
    auto mode_index = mode_for_cnst[i];
    auto aero_id = aero_for_cnst[i];
    auto gas_id = gas_for_cnst[i];
    if (gas_id != NoGas) { // constituent is a gas
      int g = static_cast<int>(gas_id);
      const Real mw = mam4::gas_species(g).molecular_weight;
      q[i] = mam4::conversions::mmr_from_vmr(vmr[i], mw);
      qqcw[i] = mam4::conversions::mmr_from_vmr(vmrcw[i], mw);
    } else {
      if (aero_id != NoAero) { // constituent is an aerosol species
        int a = aerosol_index_for_mode(mode_index, aero_id);
        const Real mw = mam4::aero_species(a).molecular_weight;
        q[i] = mam4::conversions::mmr_from_vmr(vmr[i], mw);
        qqcw[i] = mam4::conversions::mmr_from_vmr(vmrcw[i], mw);
      } else { // constituent is a modal number mixing ratio
        q[i] = vmr[i];
        qqcw[i] = vmrcw[i];
      }
    }
  }
}

#undef DECLARE_PROG_TRANSFER_CONSTANTS

} // end namespace mam4::utils

#endif
