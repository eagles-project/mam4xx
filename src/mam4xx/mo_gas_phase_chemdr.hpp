#ifndef MAM4XX_MICROPHYSICS_GAS_PHASE_CHEM_DR_HPP
#define MAM4XX_MICROPHYSICS_GAS_PHASE_CHEM_DR_HPP

#include "aero_model.hpp"
#include "compute_o3_column_density.hpp"
#include "diagnostic_arrays.hpp"
#include "gas_phase_chemistry.hpp"
#include "lin_strat_chem.hpp"
#include "mam4_amicphys.hpp"
#include "mo_drydep.hpp"
#include "mo_photo.hpp"
#include "mo_setext.hpp"
#include "mo_sethet.hpp"
#include "mo_setinv.hpp"
#include "mo_setsox.hpp"
#include "seq_drydep.hpp"

namespace mam4 {

namespace microphysics {
// number of species with external forcing
using mam4::gas_chemistry::extcnt;
using mam4::mo_photo::PhotoTableData;
using mam4::mo_setext::Forcing;
using mam4::mo_setinv::num_tracer_cnst;

using View2D = DeviceType::view_2d<Real>;
using ConstView2D = DeviceType::view_2d<const Real>;
using View1D = DeviceType::view_1d<Real>;
using ConstView1D = DeviceType::view_1d<const Real>;

// climatology data for linear stratospheric chemistry
// Per-column data is read from an NC file and then horizontally and vertically
// interpolated to the simulation grid.
struct LinozData {
  // ozone (climatology) [vmr]
  View1D linoz_o3_clim_icol;
  // temperature (climatology) [K]
  View1D linoz_t_clim_icol;
  // column o3 above box (climatology) [Dobson Units (DU)]
  View1D linoz_o3col_clim_icol;
  // P minus L (climatology) [vmr/s]
  View1D linoz_PmL_clim_icol;
  // sensitivity of P minus L to O3 [1/s]
  View1D linoz_dPmL_dO3_icol;
  // sensitivity of P minus L to T3 [K]
  View1D linoz_dPmL_dT_icol;
  // sensitivity of P minus L to overhead O3 column [vmr/DU]
  View1D linoz_dPmL_dO3col_icol;
  // Cariolle parameter for PSC loss of ozone [1/s]
  View1D linoz_cariolle_pscs_icol;
};

// configuration parameters from linoz.
// These parameters are from the namelist, except for chlorine_loading, which is
// read from an NC file.
struct LinozConf {
  Real chlorine_loading;
  // turn on/off linoz computation.
  bool compute;
  // PSC ozone loss T (K) threshold  // set from namelist input
  // linoz_psc_T
  Real psc_T;
  // number of layers with ozone decay from the surface
  int o3_lbl;
  // set from namelist input linoz_sfc
  Real o3_sfc;
  // set from namelist input linoz_tau
  Real o3_tau;
};
} // namespace microphysics
} // namespace mam4
#endif
