// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/aging.hpp>
#include <mam4xx/coagulation.hpp>
#include <mam4xx/gasaerexch.hpp>
#include <mam4xx/mam4.hpp>
#include <mam4xx/nucleation.hpp>

#include <catch2/catch.hpp>
#include <ekat_comm.hpp>
#include <ekat_logger.hpp>
#include <ekat_pack_kokkos.hpp>

#include <type_traits>

using namespace haero;
using namespace mam4;

namespace {
static constexpr int nqtendaa = 5;
static constexpr int nqtendbb = 4;
static constexpr int nqqcwtendaa = 1;
static constexpr int nqqcwtendbb = 1;
static constexpr int iqtend_cond = 0;
static constexpr int iqtend_rnam = 1;
static constexpr int iqtend_nnuc = 2;
static constexpr int iqtend_coag = 3;
static constexpr int iqtend_cond_only = 4;
static constexpr int iqqcwtend_rnam = 0;
static constexpr int maxsubarea = 2;
const Real fcvt_gas[AeroConfig::num_gas_ids()] = {1, 1, 1, 1, 1, 1};
const Real fcvt_aer[AeroConfig::num_aerosol_ids()] = {1, 1, 1, 1, 1, 1, 1};
// leave number mix-ratios unchanged (#/kmol-air)
const Real fcvt_num = 1.0;
// factor for converting aerosol water mix-ratios from (kg/kg) to (mol/mol)
const Real fcvt_wtr = 1.0;
// controls treatment of h2so4 condensation in mam_gasaerexch_1subarea
//    1 = sequential   calc. of gas-chem prod then condensation loss
//    2 = simultaneous calc. of gas-chem prod and  condensation loss
static constexpr int gaexch_h2so4_uptake_optaa = 2;

static constexpr int lmapcc_val_nul = 0;
static constexpr int lmapcc_val_gas = 1;
static constexpr int lmapcc_val_aer = 2;
static constexpr int lmapcc_val_num = 3;
static constexpr int gas_pcnst = 30;
const int lmapcc_all[gas_pcnst] = {
    lmapcc_val_nul, lmapcc_val_gas, lmapcc_val_nul, lmapcc_val_nul,
    lmapcc_val_gas, lmapcc_val_aer, lmapcc_val_aer, lmapcc_val_aer,
    lmapcc_val_aer, lmapcc_val_aer, lmapcc_val_aer, lmapcc_val_aer,
    lmapcc_val_num, lmapcc_val_aer, lmapcc_val_aer, lmapcc_val_aer,
    lmapcc_val_aer, lmapcc_val_num, lmapcc_val_aer, lmapcc_val_aer,
    lmapcc_val_aer, lmapcc_val_aer, lmapcc_val_aer, lmapcc_val_aer,
    lmapcc_val_aer, lmapcc_val_num, lmapcc_val_aer, lmapcc_val_aer,
    lmapcc_val_aer, lmapcc_val_num};

// Where lmapcc_val_num are defined in lmapcc_all
const int numptr_amode[AeroConfig::num_modes()] = {12, 17, 25, 29};
// Where lmapcc_val_gas are defined in lmapcc_all
const int lmap_gas[AeroConfig::num_modes()] = {4, 1};
// Where lmapcc_val_aer are defined in lmapcc_all
const int
    lmassptr_amode[AeroConfig::num_aerosol_ids()][AeroConfig::num_modes()] = {
        {5, 13, 18, 26}, {6, 14, 19, 27},  {7, 15, 20, 28}, {8, 16, 21, -6},
        {9, -6, 22, -6}, {10, -6, 23, -6}, {11, -6, 24, -6}};

void mam_amicphys_1subarea_clear(
    const bool do_cond, const bool do_rename, const bool do_newnuc,
    const bool do_coag, const int nstep, const Real deltat, const int jsub,
    const int nsubarea, const bool iscldy_subarea, const Real afracsub,
    const Real temp, const Real pmid, const Real pdel, const Real zmid,
    const Real pblh, const Real relhum, Real dgn_a[AeroConfig::num_modes()],
    Real dgn_awet[AeroConfig::num_modes()],
    Real wetdens[AeroConfig::num_modes()],
    const Real qgas1[AeroConfig::num_gas_ids()],
    const Real qgas3[AeroConfig::num_gas_ids()],
    Real qgas4[AeroConfig::num_gas_ids()],
    Real qgas_delaa[AeroConfig::num_gas_ids()][nqtendaa],
    const Real qnum3[AeroConfig::num_modes()],
    Real qnum4[AeroConfig::num_modes()],
    Real qnum_delaa[AeroConfig::num_modes()][nqtendaa],
    const Real qaer3[AeroConfig::num_aerosol_ids()][AeroConfig::num_modes()],
    Real qaer4[AeroConfig::num_aerosol_ids()][AeroConfig::num_modes()],
    Real qaer_delaa[AeroConfig::num_aerosol_ids()][AeroConfig::num_modes()]
                   [nqtendaa],
    const Real qwtr3[AeroConfig::num_modes()],
    Real qwtr4[AeroConfig::num_modes()]) {
  static constexpr int num_gas_ids = AeroConfig::num_gas_ids();
  static constexpr int num_modes = AeroConfig::num_modes();
  static constexpr int num_aerosol_ids = AeroConfig::num_aerosol_ids();

  static constexpr int igas_h2so4 = static_cast<int>(GasId::H2SO4);
  // Turn off nh3 for now.  This is a future enhancement.
  static constexpr int igas_nh3 = -999888777; // Same as mam_refactor
  static constexpr int iaer_so4 = static_cast<int>(AeroId::SO4);
  static constexpr int iaer_pom = static_cast<int>(AeroId::POM);
  static constexpr int newnuc_h2so4_conc_optaa = 2;

  const AeroId gas_to_aer[num_gas_ids] = {AeroId::None, AeroId::None,
                                          AeroId::SO4,  AeroId::None,
                                          AeroId::None, AeroId::SOA};

  const bool l_gas_condense_to_mode[num_gas_ids][num_modes] = {
      {false, false, false, false}, {false, false, false, false},
      {true, true, true, true}, // H2SO4
      {false, false, false, false}, {false, false, false, false},
      {true, true, true, true}}; // SOAG
  enum { NA, ANAL, IMPL };
  const int eqn_and_numerics_category[num_gas_ids] = {NA, NA, ANAL,
                                                      NA, NA, IMPL};

  // air molar density (kmol/m3)
  // const Real r_universal = Constants::r_gas; // [mJ/(K mol)]
  const Real r_universal = 8.314467591; // [mJ/(mol)] as in mam_refactor
  const Real aircon = pmid / (1000 * r_universal * temp);
  const Real alnsg_aer[num_modes] = {0.58778666490211906, 0.47000362924573563,
                                     0.58778666490211906, 0.47000362924573563};
  const Real uptk_rate_factor[num_gas_ids] = {0, 0, 1.0, 0, 0, 0.81};
  // calculates changes to gas and aerosol sub-area TMRs (tracer mixing ratios)
  // qgas3, qaer3, qnum3 are the current incoming TMRs
  // qgas4, qaer4, qnum4 are the updated outgoing TMRs
  //
  // this routine calculates changes involving
  //    gas-aerosol exchange (condensation/evaporation)
  //    growth from smaller to larger modes (renaming) due to condensation
  //    new particle nucleation
  //    coagulation
  //    transfer of particles from hydrophobic modes to hydrophilic modes
  //    (aging)
  //       due to condensation and coagulation
  //
  // qXXXN (X=gas,aer,wat,num; N=1:4) are sub-area mixing ratios
  //    XXX=gas - gas species
  //    XXX=aer - aerosol mass  species (excluding water)
  //    XXX=wat - aerosol water
  //    XXX=num - aerosol number
  //    N=1 - before gas-phase chemistry
  //    N=2 - before cloud chemistry
  //    N=3 - current incoming values (before gas-aerosol exchange, newnuc,
  //    coag) N=4 - updated outgoing values (after  gas-aerosol exchange,
  //    newnuc, coag)
  //
  // qXXX_delaa are TMR changes (not tendencies)
  //    for different processes, which are used to produce history output
  // for a clear sub-area, the processes are condensation/evaporation (and
  // associated aging), renaming, coagulation, and nucleation

  Real qgas_cur[num_gas_ids];
  for (int i = 0; i < num_gas_ids; ++i)
    qgas_cur[i] = qgas3[i];
  Real qaer_cur[num_aerosol_ids][num_modes];
  for (int i = 0; i < num_aerosol_ids; ++i)
    for (int j = 0; j < num_modes; ++j)
      qaer_cur[i][j] = qaer3[i][j];

  Real qnum_cur[num_modes];
  for (int j = 0; j < num_modes; ++j)
    qnum_cur[j] = qnum3[j];
  Real qwtr_cur[num_modes];
  for (int j = 0; j < num_modes; ++j)
    qwtr_cur[j] = qwtr3[j];

  // qgas_netprod_otrproc = gas net production rate from other processes
  //    such as gas-phase chemistry and emissions (mol/mol/s)
  // this allows the condensation (gasaerexch) routine to apply production and
  // condensation loss
  //    together, which is more accurate numerically
  // NOTE - must be >= zero, as numerical method can fail when it is negative
  // NOTE - currently only the values for h2so4 and nh3 should be non-zero
  Real qgas_netprod_otrproc[num_gas_ids] = {};
  if (do_cond && gaexch_h2so4_uptake_optaa == 2) {
    for (int igas = 0; igas < num_gas_ids; ++igas) {
      if (igas == igas_h2so4 || igas == igas_nh3) {
        // if gaexch_h2so4_uptake_optaa == 2, then
        //    if qgas increases from pre-gaschem to post-cldchem,
        //       start from the pre-gaschem mix-ratio and add in the production
        //       during the integration
        //    if it decreases,
        //       start from post-cldchem mix-ratio
        // *** currently just do this for h2so4 and nh3
        qgas_netprod_otrproc[igas] = (qgas3[igas] - qgas1[igas]) / deltat;
        if (qgas_netprod_otrproc[igas] >= 0.0)
          qgas_cur[igas] = qgas1[igas];
        else
          qgas_netprod_otrproc[igas] = 0.0;
      }
    }
  }
  Real qgas_del_cond[num_gas_ids] = {};
  Real qgas_del_nnuc[num_gas_ids] = {};
  Real qgas_del_cond_only[num_gas_ids] = {};
  Real qaer_del_cond[num_aerosol_ids][num_modes] = {};
  Real qaer_del_rnam[num_aerosol_ids][num_modes] = {};
  Real qaer_del_nnuc[num_aerosol_ids][num_modes] = {};
  Real qaer_del_coag[num_aerosol_ids][num_modes] = {};
  Real qaer_delsub_coag_in[num_aerosol_ids][AeroConfig::max_agepair()] = {};
  Real qaer_delsub_cond[num_aerosol_ids][num_modes] = {};
  Real qaer_delsub_coag[num_aerosol_ids][num_modes] = {};
  Real qaer_del_cond_only[num_aerosol_ids][num_modes] = {};
  Real qnum_del_cond[num_modes] = {};
  Real qnum_del_rnam[num_modes] = {};
  Real qnum_del_nnuc[num_modes] = {};
  Real qnum_del_coag[num_modes] = {};
  Real qnum_delsub_cond[num_modes] = {};
  Real qnum_delsub_coag[num_modes] = {};
  Real qnum_del_cond_only[num_modes] = {};
  Real dnclusterdt = 0.0;

  const int ntsubstep = 1;
  Real dtsubstep = deltat;
  if (ntsubstep > 1)
    dtsubstep = deltat / ntsubstep;
  Real del_h2so4_gasprod =
      haero::max(qgas3[igas_h2so4] - qgas1[igas_h2so4], 0.0) / ntsubstep;

  // loop over multiple time sub-steps
  for (int jtsubstep = 1; jtsubstep <= ntsubstep; ++jtsubstep) {
    // gas-aerosol exchange
    Real uptkrate_h2so4 = 0.0;
    Real del_h2so4_aeruptk = 0.0;
    Real qaer_delsub_grow4rnam[num_aerosol_ids][num_modes] = {};
    Real qgas_avg[num_gas_ids] = {};
    Real qnum_sv1[num_modes] = {};
    Real qaer_sv1[num_aerosol_ids][num_modes] = {};
    Real qgas_sv1[num_gas_ids] = {};

    if (do_cond) {

      const bool l_calc_gas_uptake_coeff = jtsubstep == 1;
      Real uptkaer[num_gas_ids][num_modes] = {};

      for (int i = 0; i < num_gas_ids; ++i)
        qgas_sv1[i] = qgas_cur[i];
      for (int i = 0; i < num_modes; ++i)
        qnum_sv1[i] = qnum_cur[i];
      for (int j = 0; j < num_aerosol_ids; ++j)
        for (int i = 0; i < num_modes; ++i)
          qaer_sv1[j][i] = qaer_cur[j][i];

      // time sub-step
      const Real dtsub_soa_fixed = -1.0;
      // Integration order
      const int nghq = 2;
      const int ntot_soamode = 4;
      int niter_out = 0;
      Real g0_soa_out = 0;
      const bool use_nh3 = false;
      gasaerexch::mam_gasaerexch_1subarea(
          nghq, igas_h2so4, use_nh3, ntot_soamode, gas_to_aer, iaer_so4,
          iaer_pom, l_calc_gas_uptake_coeff, l_gas_condense_to_mode,
          eqn_and_numerics_category, dtsubstep, dtsub_soa_fixed, temp, pmid,
          aircon, num_gas_ids, qgas_cur, qgas_avg, qgas_netprod_otrproc,
          qaer_cur, qnum_cur, dgn_awet, alnsg_aer, uptk_rate_factor, uptkaer,
          uptkrate_h2so4, niter_out, g0_soa_out);

      if (newnuc_h2so4_conc_optaa == 11)
        qgas_avg[igas_h2so4] =
            0.5 * (qgas_sv1[igas_h2so4] + qgas_cur[igas_h2so4]);
      else if (newnuc_h2so4_conc_optaa == 12)
        qgas_avg[igas_h2so4] = qgas_cur[igas_h2so4];

      for (int i = 0; i < num_gas_ids; ++i)
        qgas_del_cond[i] +=
            (qgas_cur[i] - (qgas_sv1[i] + qgas_netprod_otrproc[i] * dtsubstep));

      for (int i = 0; i < num_modes; ++i)
        qnum_delsub_cond[i] = qnum_cur[i] - qnum_sv1[i];
      for (int i = 0; i < num_aerosol_ids; ++i)
        for (int j = 0; j < num_modes; ++j)
          qaer_delsub_cond[i][j] = qaer_cur[i][j] - qaer_sv1[i][j];

      // qaer_del_grow4rnam = change in qaer_del_cond during latest condensation
      // calculations
      for (int i = 0; i < num_aerosol_ids; ++i)
        for (int j = 0; j < num_modes; ++j)
          qaer_delsub_grow4rnam[i][j] = qaer_cur[i][j] - qaer_sv1[i][j];
      for (int i = 0; i < num_gas_ids; ++i)
        qgas_del_cond_only[i] = qgas_del_cond[i];
      for (int i = 0; i < num_aerosol_ids; ++i)
        for (int j = 0; j < num_modes; ++j)
          qaer_del_cond_only[i][j] = qaer_delsub_cond[i][j];
      for (int i = 0; i < num_modes; ++i)
        qnum_del_cond_only[i] = qnum_delsub_cond[i];
      del_h2so4_aeruptk =
          qgas_cur[igas_h2so4] -
          (qgas_sv1[igas_h2so4] + qgas_netprod_otrproc[igas_h2so4] * dtsubstep);
    } else {
      for (int i = 0; i < num_gas_ids; ++i)
        qgas_avg[i] = qgas_cur[i];
    }

    // renaming after "continuous growth"
    if (do_rename) {
      constexpr int nmodes = AeroConfig::num_modes();
      constexpr int naerosol_species = AeroConfig::num_aerosol_ids();
      const Real smallest_dryvol_value = 1.0e-25; // FIXME: BAD_CONSTANT
      const int dest_mode_of_mode[nmodes] = {-1, 0, -1, -1};

      Real qnumcw_cur[num_modes] = {};
      Real qaercw_cur[num_aerosol_ids][num_modes] = {};
      Real qaercw_delsub_grow4rnam[num_aerosol_ids][num_modes] = {};
      Real mean_std_dev[nmodes];
      Real fmode_dist_tail_fac[nmodes];
      Real v2n_lo_rlx[nmodes];
      Real v2n_hi_rlx[nmodes];
      Real ln_diameter_tail_fac[nmodes];
      int num_pairs = 0;
      Real diameter_cutoff[nmodes];
      Real ln_dia_cutoff[nmodes];
      Real diameter_threshold[nmodes];
      Real mass_2_vol[naerosol_species] = {0.15,
                                           6.4971751412429377e-002,
                                           0.15,
                                           7.0588235294117650e-003,
                                           3.0789473684210526e-002,
                                           5.1923076923076926e-002,
                                           156.20986883198000};

      rename::find_renaming_pairs(dest_mode_of_mode,    // in
                                  mean_std_dev,         // out
                                  fmode_dist_tail_fac,  // out
                                  v2n_lo_rlx,           // out
                                  v2n_hi_rlx,           // out
                                  ln_diameter_tail_fac, // out
                                  num_pairs,            // out
                                  diameter_cutoff,      // out
                                  ln_dia_cutoff, diameter_threshold);

      for (int i = 0; i < num_modes; ++i)
        qnum_sv1[i] = qnum_cur[i];
      for (int j = 0; j < num_aerosol_ids; ++j)
        for (int i = 0; i < num_modes; ++i)
          qaer_sv1[j][i] = qaer_cur[j][i];
      Real dgnum_amode[nmodes];
      for (int m = 0; m < nmodes; ++m) {
        dgnum_amode[m] = modes(m).nom_diameter;
      }

      {
        Real qmol_i_cur[num_modes][num_aerosol_ids];
        Real qmol_i_del[num_modes][num_aerosol_ids];
        Real qmol_c_cur[num_modes][num_aerosol_ids];
        Real qmol_c_del[num_modes][num_aerosol_ids];
        for (int j = 0; j < num_aerosol_ids; ++j)
          for (int i = 0; i < num_modes; ++i) {
            qmol_i_cur[i][j] = qaer_cur[j][i];
            qmol_i_del[i][j] = qaer_delsub_grow4rnam[j][i];
            qmol_c_cur[i][j] = qaercw_cur[j][i];
            qmol_c_del[i][j] = qaercw_delsub_grow4rnam[j][i];
          }
        Rename rename;
        rename.mam_rename_1subarea_(
            iscldy_subarea, smallest_dryvol_value, dest_mode_of_mode,
            mean_std_dev, fmode_dist_tail_fac, v2n_lo_rlx, v2n_hi_rlx,
            ln_diameter_tail_fac, num_pairs, diameter_cutoff, ln_dia_cutoff,
            diameter_threshold, mass_2_vol, dgnum_amode, qnum_cur, qmol_i_cur,
            qmol_i_del, qnumcw_cur, qmol_c_cur, qmol_c_del);

        for (int j = 0; j < num_aerosol_ids; ++j)
          for (int i = 0; i < num_modes; ++i) {
            qaer_cur[j][i] = qmol_i_cur[i][j];
            qaer_delsub_grow4rnam[j][i] = qmol_i_del[i][j];
            qaercw_cur[j][i] = qmol_c_cur[i][j];
            qaercw_delsub_grow4rnam[j][i] = qmol_c_del[i][j];
          }
      }

      for (int i = 0; i < num_modes; ++i)
        qnum_del_rnam[i] += qnum_cur[i] - qnum_sv1[i];
      for (int i = 0; i < num_aerosol_ids; ++i)
        for (int j = 0; j < num_modes; ++j)
          qaer_del_rnam[i][j] += qaer_cur[i][j] - qaer_sv1[i][j];
    }

    // new particle formation (nucleation)
    if (do_newnuc) {
      for (int i = 0; i < num_gas_ids; ++i)
        qgas_sv1[i] = qgas_cur[i];
      for (int i = 0; i < num_modes; ++i)
        qnum_sv1[i] = qnum_cur[i];
      Real qaer_cur_tmp[num_modes][num_aerosol_ids];
      for (int j = 0; j < num_aerosol_ids; ++j)
        for (int i = 0; i < num_modes; ++i) {
          qaer_sv1[j][i] = qaer_cur[j][i];
          qaer_cur_tmp[i][j] = qaer_cur[j][i];
        }
      int nsize = 1;
      int isize_nuc = 1;
      Real qnuma_del;
      Real qso4a_del;
      Real qnh4a_del;
      Real qh2so4_del;
      Real qnh3_del;
      Real dnclusterdt_substep = 0;
      Real dndt_ait = 0;
      Real dmdt_ait = 0;
      Real dso4dt_ait = 0;
      Real dnh4dt_ait = 0;
      Nucleation nucleation;
      Nucleation::Config config;
      Real dens_so4a_host = 1770;
      config.mw_nh4a_host = 115;
      config.mw_so4a_host = 115;
      // config.accom_coef_h2so4 = 0.65;
      AeroConfig aero_config;
      nucleation.init(aero_config, config);
      Real dplom_mode = nucleation.get_dp_lo_mode();
      Real dphim_mode = nucleation.get_dp_hi_mode();
      // old call to box-model version
      // nucleation.compute_tendencies_(
      //     dtsubstep, temp, pmid, aircon, zmid, pblh, relhum, uptkrate_h2so4,
      //     del_h2so4_gasprod, del_h2so4_aeruptk, qgas_cur, qgas_avg, qnum_cur,
      //     qaer_cur_tmp, qwtr_cur, dndt_ait, dmdt_ait, dso4dt_ait, dnh4dt_ait,
      //     dnclusterdt_substep);
      nucleation.unit_testing_compute_tendencies_(
          dtsubstep, temp, pmid, zmid, pblh, relhum, uptkrate_h2so4, nsize,
          dplom_mode, dphim_mode, qgas_cur, qgas_avg, dens_so4a_host, isize_nuc,
          qnuma_del, qh2so4_del, qnh4a_del, qso4a_del, qnh3_del, dndt_ait,
          dmdt_ait, dso4dt_ait, dnh4dt_ait, dnclusterdt_substep);
      for (int j = 0; j < num_aerosol_ids; ++j)
        for (int i = 0; i < num_modes; ++i)
          qaer_cur[j][i] = qaer_cur_tmp[i][j];

      std::cout << "qso4a_del = " << qso4a_del << "\n";
      std::cout << "qh2so4_del = " << qh2so4_del << "\n";

      // Apply the tendencies to the prognostics.
      const int nait = static_cast<int>(ModeIndex::Aitken);
      qnum_cur[nait] += dndt_ait * dtsubstep;

      if (dso4dt_ait > 0.0) {
        static constexpr int iaer_so4 = static_cast<int>(AeroId::SO4);
        static constexpr int igas_h2so4 = static_cast<int>(GasId::H2SO4);

        Real delta_q = dso4dt_ait * dtsubstep;
        qaer_cur[iaer_so4][nait] += delta_q;
        delta_q = haero::min(delta_q, qgas_cur[igas_h2so4]);
        qgas_cur[igas_h2so4] -= delta_q;
      }

      if (igas_nh3 > 0 && dnh4dt_ait > 0.0) {
        static constexpr int iaer_nh4 =
            -9999999; // static_cast<int>(AeroId::NH4);

        Real delta_q = dnh4dt_ait * dtsubstep;
        qaer_cur[iaer_nh4][nait] += delta_q;
        delta_q = haero::min(delta_q, qgas_cur[igas_nh3]);
        qgas_cur[igas_nh3] -= delta_q;
      }
      for (int i = 0; i < num_gas_ids; ++i)
        qgas_del_nnuc[i] += (qgas_cur[i] - qgas_sv1[i]);
      for (int i = 0; i < num_modes; ++i)
        qnum_del_nnuc[i] += (qnum_cur[i] - qnum_sv1[i]);
      for (int j = 0; j < num_aerosol_ids; ++j)
        for (int i = 0; i < num_modes; ++i)
          qaer_del_nnuc[j][i] += (qaer_cur[j][i] - qaer_sv1[j][i]);

      dnclusterdt = dnclusterdt + dnclusterdt_substep * (dtsubstep / deltat);
    }

    // coagulation part
    if (do_coag) {
      for (int i = 0; i < num_modes; ++i)
        qnum_sv1[i] = qnum_cur[i];
      for (int j = 0; j < num_aerosol_ids; ++j)
        for (int i = 0; i < num_modes; ++i)
          qaer_sv1[j][i] = qaer_cur[j][i];
      coagulation::mam_coag_1subarea(dtsubstep, temp, pmid, aircon, dgn_a,
                                     dgn_awet, wetdens, qnum_cur, qaer_cur,
                                     qaer_delsub_coag_in);
      for (int i = 0; i < num_modes; ++i)
        qnum_delsub_coag[i] = qnum_cur[i] - qnum_sv1[i];
      for (int j = 0; j < num_aerosol_ids; ++j)
        for (int i = 0; i < num_modes; ++i)
          qaer_delsub_coag[j][i] = qaer_cur[j][i] - qaer_sv1[j][i];
    }

    // primary carbon aging

    aging::mam_pcarbon_aging_1subarea(
        dgn_a, qnum_cur, qnum_delsub_cond, qnum_delsub_coag, qaer_cur,
        qaer_delsub_cond, qaer_delsub_coag, qaer_delsub_coag_in);

    // accumulate sub-step q-dels
    if (do_coag) {
      for (int i = 0; i < num_modes; ++i)
        qnum_del_coag[i] += qnum_delsub_coag[i];
      for (int j = 0; j < num_aerosol_ids; ++j)
        for (int i = 0; i < num_modes; ++i)
          qaer_del_coag[j][i] += qaer_delsub_coag[j][i];
    }
    if (do_cond) {
      for (int i = 0; i < num_modes; ++i)
        qnum_del_cond[i] += qnum_delsub_cond[i];
      for (int j = 0; j < num_aerosol_ids; ++j)
        for (int i = 0; i < num_modes; ++i)
          qaer_del_cond[j][i] += qaer_delsub_cond[j][i];
    }
  }

  // final mix ratios
  for (int i = 0; i < num_gas_ids; ++i)
    qgas4[i] = qgas_cur[i];
  for (int j = 0; j < num_aerosol_ids; ++j)
    for (int i = 0; i < num_modes; ++i)
      qaer4[j][i] = qaer_cur[j][i];
  for (int i = 0; i < num_modes; ++i)
    qnum4[i] = qnum_cur[i];
  for (int i = 0; i < num_modes; ++i)
    qwtr4[i] = qwtr_cur[i];

  // final mix ratio changes
  for (int i = 0; i < num_gas_ids; ++i) {
    qgas_delaa[i][iqtend_cond] = qgas_del_cond[i];
    qgas_delaa[i][iqtend_rnam] = 0.0;
    qgas_delaa[i][iqtend_nnuc] = qgas_del_nnuc[i];
    qgas_delaa[i][iqtend_coag] = 0.0;
    qgas_delaa[i][iqtend_cond_only] = qgas_del_cond_only[i];
  }
  for (int i = 0; i < num_modes; ++i) {
    qnum_delaa[i][iqtend_cond] = qnum_del_cond[i];
    qnum_delaa[i][iqtend_rnam] = qnum_del_rnam[i];
    qnum_delaa[i][iqtend_nnuc] = qnum_del_nnuc[i];
    qnum_delaa[i][iqtend_coag] = qnum_del_coag[i];
    qnum_delaa[i][iqtend_cond_only] = qnum_del_cond_only[i];
  }
  for (int j = 0; j < num_aerosol_ids; ++j) {
    for (int i = 0; i < num_modes; ++i) {
      qaer_delaa[j][i][iqtend_cond] = qaer_del_cond[j][i];
      qaer_delaa[j][i][iqtend_rnam] = qaer_del_rnam[j][i];
      qaer_delaa[j][i][iqtend_nnuc] = qaer_del_nnuc[j][i];
      qaer_delaa[j][i][iqtend_coag] = qaer_del_coag[j][i];
      qaer_delaa[j][i][iqtend_cond_only] = qaer_del_cond_only[j][i];
    }
  }
}

// -----------------------------------------------------------------------
// -----------------------------------------------------------------------

void mam_amicphys_1subarea_cloudy(
    const bool do_cond, const bool do_rename, const bool do_newnuc,
    const bool do_coag, const int nstep, const Real deltat, const int jsub,
    const int nsubarea, const bool iscldy_subarea, const Real afracsub,
    const Real temp, const Real pmid, const Real pdel, const Real zmid,
    const Real pblh, const Real relhum, Real dgn_a[AeroConfig::num_modes()],
    Real dgn_awet[AeroConfig::num_modes()],
    Real wetdens[AeroConfig::num_modes()],
    const Real qgas1[AeroConfig::num_gas_ids()],
    const Real qgas3[AeroConfig::num_gas_ids()],
    Real qgas4[AeroConfig::num_gas_ids()],
    Real qgas_delaa[AeroConfig::num_gas_ids()][nqtendaa],
    const Real qnum3[AeroConfig::num_modes()],
    Real qnum4[AeroConfig::num_modes()],
    Real qnum_delaa[AeroConfig::num_modes()][nqtendaa],
    const Real qaer2[AeroConfig::num_aerosol_ids()][AeroConfig::num_modes()],
    const Real qaer3[AeroConfig::num_aerosol_ids()][AeroConfig::num_modes()],
    Real qaer4[AeroConfig::num_aerosol_ids()][AeroConfig::num_modes()],
    Real qaer_delaa[AeroConfig::num_aerosol_ids()][AeroConfig::num_modes()]
                   [nqtendaa],
    const Real qwtr3[AeroConfig::num_modes()],
    Real qwtr4[AeroConfig::num_modes()],
    const Real qnumcw3[AeroConfig::num_modes()],
    Real qnumcw4[AeroConfig::num_modes()],
    Real qnumcw_delaa[AeroConfig::num_modes()][nqqcwtendaa],
    const Real qaercw2[AeroConfig::num_gas_ids()][AeroConfig::num_modes()],
    const Real qaercw3[AeroConfig::num_gas_ids()][AeroConfig::num_modes()],
    Real qaercw4[AeroConfig::num_gas_ids()][AeroConfig::num_modes()],
    Real qaercw_delaa[AeroConfig::num_gas_ids()][AeroConfig::num_modes()]
                     [nqqcwtendaa]) {

  //
  // calculates changes to gas and aerosol sub-area TMRs (tracer mixing ratios)
  // qgas3, qaer3, qaercw3, qnum3, qnumcw3 are the current incoming TMRs
  // qgas4, qaer4, qaercw4, qnum4, qnumcw4 are the updated outgoing TMRs
  //
  // when do_cond = false, this routine only calculates changes involving
  //    growth from smaller to larger modes (renaming) following cloud chemistry
  //    so gas TMRs are not changed
  // when do_cond = true, this routine also calculates changes involving
  //    gas-aerosol exchange (condensation/evaporation)
  //    transfer of particles from hydrophobic modes to hydrophilic modes
  //    (aging)
  //       due to condensation
  // currently this routine does not do
  //    new particle nucleation - because h2so4 gas conc. should be very low in
  //    cloudy air coagulation - because cloud-borne aerosol would need to be
  //    included
  //

  // qXXXN (X=gas,aer,wat,num; N=1:4) are sub-area mixing ratios
  //    XXX=gas - gas species
  //    XXX=aer - aerosol mass  species (excluding water)
  //    XXX=wat - aerosol water
  //    XXX=num - aerosol number
  //    N=1 - before gas-phase chemistry
  //    N=2 - before cloud chemistry
  //    N=3 - current incoming values (before gas-aerosol exchange, newnuc,
  //    coag) N=4 - updated outgoing values (after  gas-aerosol exchange,
  //    newnuc, coag)
  //
  // qXXX_delaa are TMR changes (not tendencies)
  //    for different processes, which are used to produce history output
  // for a clear sub-area, the processes are condensation/evaporation (and
  // associated aging),
  //    renaming, coagulation, and nucleation

  // qxxx_del_yyyy    are mix-ratio changes over full time step (deltat)
  // qxxx_delsub_yyyy are mix-ratio changes over time sub-step (dtsubstep)

  static constexpr int num_gas_ids = AeroConfig::num_gas_ids();
  static constexpr int num_modes = AeroConfig::num_modes();
  static constexpr int num_aerosol_ids = AeroConfig::num_aerosol_ids();

  static constexpr int igas_h2so4 = static_cast<int>(GasId::H2SO4);
  // Turn off nh3 for now.  This is a future enhancement.
  static constexpr int igas_nh3 = -999888777; // Same as mam_refactor
  static constexpr int iaer_so4 = static_cast<int>(AeroId::SO4);
  static constexpr int iaer_pom = static_cast<int>(AeroId::POM);
  static constexpr int newnuc_h2so4_conc_optaa = 2;

  const AeroId gas_to_aer[num_gas_ids] = {AeroId::None, AeroId::None,
                                          AeroId::SO4,  AeroId::None,
                                          AeroId::None, AeroId::SOA};
  const bool l_gas_condense_to_mode[num_gas_ids][num_modes] = {
      {false, false, false, false}, {false, false, false, false},
      {true, true, true, true}, // H2SO4
      {false, false, false, false}, {false, false, false, false},
      {true, true, true, true}}; // SOAG
  enum { NA, ANAL, IMPL };
  const int eqn_and_numerics_category[num_gas_ids] = {NA, NA, ANAL,
                                                      NA, NA, IMPL};
  // air molar density (kmol/m3)
  // In order to try to match the results in mam_refactor
  // set r_universal as  [mJ/(mol)] as in mam_refactor.
  // const Real r_universal = Constants::r_gas; // [mJ/(K mol)]
  const Real r_universal = 8.314467591; // [mJ/(mol)] as in mam_refactor
  const Real aircon = pmid / (1000 * r_universal * temp);
  const Real alnsg_aer[num_modes] = {0.58778666490211906, 0.47000362924573563,
                                     0.58778666490211906, 0.47000362924573563};
  const Real uptk_rate_factor[num_gas_ids] = {0, 0, 1.0, 0, 0, 0.81};

  Real qgas_cur[num_gas_ids];
  for (int i = 0; i < num_gas_ids; ++i)
    qgas_cur[i] = qgas3[i];
  Real qaer_cur[num_aerosol_ids][num_modes];
  for (int i = 0; i < num_aerosol_ids; ++i)
    for (int j = 0; j < num_modes; ++j)
      qaer_cur[i][j] = qaer3[i][j];

  Real qnum_cur[num_modes];
  for (int j = 0; j < num_modes; ++j)
    qnum_cur[j] = qnum3[j];
  Real qwtr_cur[num_modes];
  for (int j = 0; j < num_modes; ++j)
    qwtr_cur[j] = qwtr3[j];

  Real qnumcw_cur[num_modes];
  for (int j = 0; j < num_modes; ++j)
    qnumcw_cur[j] = qnumcw3[j];

  Real qaercw_cur[num_gas_ids][num_modes];
  for (int i = 0; i < num_gas_ids; ++i)
    for (int j = 0; j < num_modes; ++j)
      qaercw_cur[i][j] = qaercw3[i][j];

  Real qgas_netprod_otrproc[num_gas_ids] = {};
  if (do_cond && gaexch_h2so4_uptake_optaa == 2) {
    for (int igas = 0; igas < num_gas_ids; ++igas) {
      if (igas == igas_h2so4 || igas == igas_nh3) {
        // if gaexch_h2so4_uptake_optaa == 2, then
        //    if qgas increases from pre-gaschem to post-cldchem,
        //       start from the pre-gaschem mix-ratio and add in the production
        //       during the integration
        //    if it decreases,
        //       start from post-cldchem mix-ratio
        // *** currently just do this for h2so4 and nh3
        qgas_netprod_otrproc[igas] = (qgas3[igas] - qgas1[igas]) / deltat;
        if (qgas_netprod_otrproc[igas] >= 0.0)
          qgas_cur[igas] = qgas1[igas];
        else
          qgas_netprod_otrproc[igas] = 0.0;
      }
    }
  }
  Real qgas_del_cond[num_gas_ids] = {};
  Real qgas_del_nnuc[num_gas_ids] = {};
  Real qgas_del_cond_only[num_gas_ids] = {};
  Real qaer_del_cond[num_aerosol_ids][num_modes] = {};
  Real qaer_del_rnam[num_aerosol_ids][num_modes] = {};
  Real qaer_del_nnuc[num_aerosol_ids][num_modes] = {};
  Real qaer_del_coag[num_aerosol_ids][num_modes] = {};
  Real qaer_delsub_cond[num_aerosol_ids][num_modes] = {};
  Real qaer_delsub_coag[num_aerosol_ids][num_modes] = {};
  Real qaer_del_cond_only[num_aerosol_ids][num_modes] = {};
  Real qaercw_del_rnam[num_aerosol_ids][num_modes] = {};
  Real qnum_del_cond[num_modes] = {};
  Real qnum_del_rnam[num_modes] = {};
  Real qnum_del_nnuc[num_modes] = {};
  Real qnum_del_coag[num_modes] = {};
  Real qnum_delsub_cond[num_modes] = {};
  Real qnum_delsub_coag[num_modes] = {};
  Real qnum_del_cond_only[num_modes] = {};
  Real qnumcw_del_rnam[num_modes] = {};
  Real qaer_delsub_coag_in[num_aerosol_ids][AeroConfig::max_agepair()] = {};

  const int ntsubstep = 1;
  Real dtsubstep = deltat;
  if (ntsubstep > 1)
    dtsubstep = deltat / ntsubstep;

  // loop over multiple time sub-steps

  for (int jtsubstep = 1; jtsubstep <= ntsubstep; ++jtsubstep) {
    // gas-aerosol exchange
    Real uptkrate_h2so4 = 0.0;
    Real qgas_avg[num_gas_ids] = {};
    Real qgas_sv1[num_gas_ids] = {};
    Real qnum_sv1[num_modes] = {};
    Real qaer_sv1[num_aerosol_ids][num_modes] = {};
    Real qaer_delsub_grow4rnam[num_aerosol_ids][num_modes] = {};

    if (do_cond) {

      const bool l_calc_gas_uptake_coeff = jtsubstep == 1;
      Real uptkaer[num_gas_ids][num_modes] = {};

      for (int i = 0; i < num_gas_ids; ++i)
        qgas_sv1[i] = qgas_cur[i];
      for (int i = 0; i < num_modes; ++i)
        qnum_sv1[i] = qnum_cur[i];
      for (int j = 0; j < num_aerosol_ids; ++j)
        for (int i = 0; i < num_modes; ++i)
          qaer_sv1[j][i] = qaer_cur[j][i];

      const int nghq = 2;
      const int ntot_soamode = 4;
      int niter_out = 0;
      Real g0_soa_out = 0;
      // time sub-step
      const Real dtsub_soa_fixed = -1.0;
      const bool use_nh3 = false;
      gasaerexch::mam_gasaerexch_1subarea(
          nghq, igas_h2so4, use_nh3, ntot_soamode, gas_to_aer, iaer_so4,
          iaer_pom, l_calc_gas_uptake_coeff, l_gas_condense_to_mode,
          eqn_and_numerics_category, dtsubstep, dtsub_soa_fixed, temp, pmid,
          aircon, num_gas_ids, qgas_cur, qgas_avg, qgas_netprod_otrproc,
          qaer_cur, qnum_cur, dgn_awet, alnsg_aer, uptk_rate_factor, uptkaer,
          uptkrate_h2so4, niter_out, g0_soa_out);

      if (newnuc_h2so4_conc_optaa == 11)
        qgas_avg[igas_h2so4] =
            0.5 * (qgas_sv1[igas_h2so4] + qgas_cur[igas_h2so4]);
      else if (newnuc_h2so4_conc_optaa == 12)
        qgas_avg[igas_h2so4] = qgas_cur[igas_h2so4];

      for (int i = 0; i < num_gas_ids; ++i)
        qgas_del_cond[i] +=
            (qgas_cur[i] - (qgas_sv1[i] + qgas_netprod_otrproc[i] * dtsubstep));

      for (int i = 0; i < num_modes; ++i)
        qnum_delsub_cond[i] = qnum_cur[i] - qnum_sv1[i];
      for (int i = 0; i < num_aerosol_ids; ++i)
        for (int j = 0; j < num_modes; ++j)
          qaer_delsub_cond[i][j] = qaer_cur[i][j] - qaer_sv1[i][j];

      // qaer_del_grow4rnam = change in qaer_del_cond during latest condensation
      // calculations
      for (int i = 0; i < num_aerosol_ids; ++i)
        for (int j = 0; j < num_modes; ++j)
          qaer_delsub_grow4rnam[i][j] = qaer_cur[i][j] - qaer_sv1[i][j];
      for (int i = 0; i < num_gas_ids; ++i)
        qgas_del_cond_only[i] = qgas_del_cond[i];
      for (int i = 0; i < num_aerosol_ids; ++i)
        for (int j = 0; j < num_modes; ++j)
          qaer_del_cond_only[i][j] = qaer_delsub_cond[i][j];
      for (int i = 0; i < num_modes; ++i)
        qnum_del_cond_only[i] = qnum_delsub_cond[i];

    } else {
      for (int i = 0; i < num_gas_ids; ++i)
        qgas_avg[i] = qgas_cur[i];
    }
    // renaming after "continuous growth"
    if (do_rename) {
      constexpr int nmodes = AeroConfig::num_modes();
      constexpr int naerosol_species = AeroConfig::num_aerosol_ids();
      const Real smallest_dryvol_value = 1.0e-25; // FIXME: BAD_CONSTANT
      const int dest_mode_of_mode[nmodes] = {-1, 0, -1, -1};

      Real qnumcw_cur[num_modes] = {};
      Real qaercw_cur[num_aerosol_ids][num_modes] = {};
      Real qaercw_delsub_grow4rnam[num_aerosol_ids][num_modes] = {};
      Real mean_std_dev[nmodes];
      Real fmode_dist_tail_fac[nmodes];
      Real v2n_lo_rlx[nmodes];
      Real v2n_hi_rlx[nmodes];
      Real ln_diameter_tail_fac[nmodes];
      int num_pairs = 0;
      Real diameter_cutoff[nmodes];
      Real ln_dia_cutoff[nmodes];
      Real diameter_threshold[nmodes];
      Real mass_2_vol[naerosol_species] = {0.15,
                                           6.4971751412429377e-002,
                                           0.15,
                                           7.0588235294117650e-003,
                                           3.0789473684210526e-002,
                                           5.1923076923076926e-002,
                                           156.20986883198000};

      rename::find_renaming_pairs(dest_mode_of_mode,    // in
                                  mean_std_dev,         // out
                                  fmode_dist_tail_fac,  // out
                                  v2n_lo_rlx,           // out
                                  v2n_hi_rlx,           // out
                                  ln_diameter_tail_fac, // out
                                  num_pairs,            // out
                                  diameter_cutoff,      // out
                                  ln_dia_cutoff, diameter_threshold);

      for (int i = 0; i < num_modes; ++i)
        qnum_sv1[i] = qnum_cur[i];
      for (int j = 0; j < num_aerosol_ids; ++j)
        for (int i = 0; i < num_modes; ++i)
          qaer_sv1[j][i] = qaer_cur[j][i];
      Real dgnum_amode[nmodes];
      for (int m = 0; m < nmodes; ++m) {
        dgnum_amode[m] = modes(m).nom_diameter;
      }

      // qaercw_delsub_grow4rnam = change in qaercw from cloud chemistry
      for (int i = 0; i < num_aerosol_ids; ++i)
        for (int j = 0; j < num_modes; ++j)
          qaercw_delsub_grow4rnam[i][j] =
              (qaercw3[i][j] - qaercw2[i][j]) / ntsubstep;
      Real qnumcw_sv1[num_modes];
      for (int i = 0; i < num_modes; ++i)
        qnumcw_sv1[i] = qnumcw_cur[i];
      Real qaercw_sv1[num_aerosol_ids][num_modes];
      for (int i = 0; i < num_aerosol_ids; ++i)
        for (int j = 0; j < num_modes; ++j)
          qaercw_sv1[i][j] = qaercw_cur[i][j];

      {
        Real qmol_i_cur[num_modes][num_aerosol_ids];
        Real qmol_i_del[num_modes][num_aerosol_ids];
        Real qmol_c_cur[num_modes][num_aerosol_ids];
        Real qmol_c_del[num_modes][num_aerosol_ids];
        for (int j = 0; j < num_aerosol_ids; ++j)
          for (int i = 0; i < num_modes; ++i) {
            qmol_i_cur[i][j] = qaer_cur[j][i];
            qmol_i_del[i][j] = qaer_delsub_grow4rnam[j][i];
            qmol_c_cur[i][j] = qaercw_cur[j][i];
            qmol_c_del[i][j] = qaercw_delsub_grow4rnam[j][i];
          }

        Rename rename;
        rename.mam_rename_1subarea_(
            iscldy_subarea, smallest_dryvol_value, dest_mode_of_mode,
            mean_std_dev, fmode_dist_tail_fac, v2n_lo_rlx, v2n_hi_rlx,
            ln_diameter_tail_fac, num_pairs, diameter_cutoff, ln_dia_cutoff,
            diameter_threshold, mass_2_vol, dgnum_amode, qnum_cur, qmol_i_cur,
            qmol_i_del, qnumcw_cur, qmol_c_cur, qmol_c_del);

        for (int j = 0; j < num_aerosol_ids; ++j)
          for (int i = 0; i < num_modes; ++i) {
            qaer_cur[j][i] = qmol_i_cur[i][j];
            qaer_delsub_grow4rnam[j][i] = qmol_i_del[i][j];
            qaercw_cur[j][i] = qmol_c_cur[i][j];
            qaercw_delsub_grow4rnam[j][i] = qmol_c_del[i][j];
          }
      }
      for (int i = 0; i < num_modes; ++i)
        qnum_del_rnam[i] += qnum_cur[i] - qnum_sv1[i];
      for (int i = 0; i < num_aerosol_ids; ++i)
        for (int j = 0; j < num_modes; ++j)
          qaer_del_rnam[i][j] += qaer_cur[i][j] - qaer_sv1[i][j];
      for (int i = 0; i < num_modes; ++i)
        qnumcw_del_rnam[i] += qnumcw_cur[i] - qnumcw_sv1[i];
      for (int i = 0; i < num_aerosol_ids; ++i)
        for (int j = 0; j < num_modes; ++j)
          qaercw_del_rnam[i][j] += qaercw_cur[i][j] - qaercw_sv1[i][j];
    }

    // primary carbon aging
    if (do_cond) {
      aging::mam_pcarbon_aging_1subarea(
          dgn_a, qnum_cur, qnum_delsub_cond, qnum_delsub_coag, qaer_cur,
          qaer_delsub_cond, qaer_delsub_coag, qaer_delsub_coag_in);
    }
    // accumulate sub-step q-dels
    if (do_cond) {
      for (int i = 0; i < num_modes; ++i)
        qnum_del_cond[i] += qnum_delsub_cond[i];
      for (int j = 0; j < num_aerosol_ids; ++j)
        for (int i = 0; i < num_modes; ++i)
          qaer_del_cond[j][i] += qaer_delsub_cond[j][i];
    }
  }
  // final mix ratios
  for (int i = 0; i < num_gas_ids; ++i)
    qgas4[i] = qgas_cur[i];
  for (int j = 0; j < num_aerosol_ids; ++j)
    for (int i = 0; i < num_modes; ++i)
      qaer4[j][i] = qaer_cur[j][i];
  for (int i = 0; i < num_modes; ++i)
    qnum4[i] = qnum_cur[i];
  for (int i = 0; i < num_modes; ++i)
    qwtr4[i] = qwtr_cur[i];
  for (int i = 0; i < num_modes; ++i)
    qnumcw4[i] = qnumcw_cur[i];
  for (int i = 0; i < num_gas_ids; ++i)
    for (int j = 0; j < num_modes; ++j)
      qaercw4[i][j] = qaercw_cur[i][j];

  // final mix ratio changes
  for (int i = 0; i < num_gas_ids; ++i) {
    qgas_delaa[i][iqtend_cond] = qgas_del_cond[i];
    qgas_delaa[i][iqtend_rnam] = 0.0;
    qgas_delaa[i][iqtend_nnuc] = qgas_del_nnuc[i];
    qgas_delaa[i][iqtend_coag] = 0.0;
    qgas_delaa[i][iqtend_cond_only] = qgas_del_cond_only[i];
  }
  for (int i = 0; i < num_modes; ++i) {
    qnum_delaa[i][iqtend_cond] = qnum_del_cond[i];
    qnum_delaa[i][iqtend_rnam] = qnum_del_rnam[i];
    qnum_delaa[i][iqtend_nnuc] = qnum_del_nnuc[i];
    qnum_delaa[i][iqtend_coag] = qnum_del_coag[i];
    qnum_delaa[i][iqtend_cond_only] = qnum_del_cond_only[i];
  }
  for (int j = 0; j < num_aerosol_ids; ++j) {
    for (int i = 0; i < num_modes; ++i) {
      qaer_delaa[j][i][iqtend_cond] = qaer_del_cond[j][i];
      qaer_delaa[j][i][iqtend_rnam] = qaer_del_rnam[j][i];
      qaer_delaa[j][i][iqtend_nnuc] = qaer_del_nnuc[j][i];
      qaer_delaa[j][i][iqtend_coag] = qaer_del_coag[j][i];
      qaer_delaa[j][i][iqtend_cond_only] = qaer_del_cond_only[j][i];
    }
  }
  for (int i = 0; i < num_modes; ++i)
    qnumcw_delaa[i][iqqcwtend_rnam] = qnumcw_del_rnam[i];
  for (int i = 0; i < num_aerosol_ids; ++i)
    for (int j = 0; j < num_modes; ++j)
      qaercw_delaa[i][j][iqqcwtend_rnam] = qaercw_del_rnam[i][j];
}

void mam_amicphys_1gridcell(
    const bool do_cond, const bool do_rename, const bool do_newnuc,
    const bool do_coag, const int nstep, const Real deltat, const int nsubarea,
    const int ncldy_subarea, const bool iscldy_subarea[maxsubarea],
    const Real afracsub[maxsubarea], const Real temp, const Real pmid,
    const Real pdel, const Real zmid, const Real pblh,
    const Real relhumsub[maxsubarea], Real dgn_a[AeroConfig::num_modes()],
    Real dgn_awet[AeroConfig::num_modes()],
    Real wetdens[AeroConfig::num_modes()],
    const Real qsub1[AeroConfig::num_gas_ids()][maxsubarea],
    const Real qsub2[AeroConfig::num_gas_ids()][maxsubarea],
    const Real qqcwsub2[AeroConfig::num_gas_ids()][maxsubarea],
    const Real qsub3[AeroConfig::num_gas_ids()][maxsubarea],
    const Real qqcwsub3[AeroConfig::num_gas_ids()][maxsubarea],
    Real qaerwatsub3[AeroConfig::num_modes()][maxsubarea],
    Real qsub4[AeroConfig::num_gas_ids()][maxsubarea],
    Real qqcwsub4[AeroConfig::num_gas_ids()][maxsubarea],
    Real qaerwatsub4[AeroConfig::num_modes()][maxsubarea],
    Real qsub_tendaa[AeroConfig::num_gas_ids()][nqtendaa][maxsubarea],
    Real qqcwsub_tendaa[AeroConfig::num_gas_ids()][nqqcwtendaa][maxsubarea]) {

  //
  // calculates changes to gas and aerosol sub-area TMRs (tracer mixing ratios)
  // qsub3 and qqcwsub3 are the incoming current TMRs
  // qsub4 and qqcwsub4 are the outgoing updated TMRs
  //
  // qsubN and qqcwsubN (N=1:4) are tracer mixing ratios (TMRs, mol/mol or
  // #/kmol) in sub-areas
  //    currently there are just clear and cloudy sub-areas
  //    the N=1:4 have same meanings as for qgcmN
  //    N=1 - before gas-phase chemistry
  //    N=2 - before cloud chemistry
  //    N=3 - incoming values (before gas-aerosol exchange, newnuc, coag)
  //    N=4 - outgoing values (after  gas-aerosol exchange, newnuc, coag)
  // qsub_tendaa and qqcwsub_tendaa are TMR tendencies
  //    for different processes, which are used to produce history output
  // the processes are condensation/evaporation (and associated aging),
  //    renaming, coagulation, and nucleation

  static constexpr int num_gas_ids = AeroConfig::num_gas_ids();
  static constexpr int num_modes = AeroConfig::num_modes();
  static constexpr int num_aerosol_ids = AeroConfig::num_aerosol_ids();

  // the q--4 values will be equal to q--3 values unless they get changed
  for (int i = 0; i < num_gas_ids; ++i)
    for (int j = 0; j < maxsubarea; ++j) {
      qsub4[i][j] = qsub3[i][j];
      qqcwsub4[i][j] = qqcwsub3[i][j];
    }
  for (int i = 0; i < num_modes; ++i)
    for (int j = 0; j < maxsubarea; ++j)
      qaerwatsub4[i][j] = qaerwatsub3[i][j];
  for (int i = 0; i < num_gas_ids; ++i)
    for (int j = 0; j < nqtendaa; ++j)
      for (int k = 0; k < maxsubarea; ++k)
        qsub_tendaa[i][j][k] = 0;
  for (int i = 0; i < num_gas_ids; ++i)
    for (int j = 0; j < nqqcwtendaa; ++j)
      for (int k = 0; k < maxsubarea; ++k)
        qqcwsub_tendaa[i][j][k] = 0.0;

  for (int jsub = 0; jsub < nsubarea; ++jsub) {
    bool do_cond_sub;
    bool do_rename_sub;
    bool do_newnuc_sub;
    bool do_coag_sub;

    if (iscldy_subarea[jsub]) {
      do_cond_sub = do_cond;
      do_rename_sub = do_rename;
      do_newnuc_sub = false;
      do_coag_sub = false;
    } else {
      do_cond_sub = do_cond;
      do_rename_sub = do_rename;
      do_newnuc_sub = do_newnuc;
      do_coag_sub = do_coag;
    }
    const bool do_map_gas_sub = do_cond_sub || do_newnuc_sub;

    // map incoming sub-area mix-ratios to gas/aer/num arrays
    Real qgas1[num_gas_ids] = {};
    Real qgas3[num_gas_ids] = {};
    Real qgas4[num_gas_ids] = {};
    if (do_map_gas_sub) {
      // for cldy subarea, only do gases if doing gaexch
      for (int igas = 0; igas < 2; ++igas) {
        const int l = lmap_gas[igas];
        qgas1[igas] = qsub1[l][jsub] * fcvt_gas[igas];
        qgas3[igas] = qsub3[l][jsub] * fcvt_gas[igas];
        qgas4[igas] = qgas3[igas];
      }
    }
    Real qaer2[num_aerosol_ids][num_modes] = {};
    Real qaer3[num_aerosol_ids][num_modes] = {};
    Real qnum3[num_modes] = {};
    Real qaer4[num_aerosol_ids][num_modes] = {};
    Real qnum4[num_modes] = {};
    Real qwtr3[num_modes] = {};
    Real qwtr4[num_modes] = {};
    for (int n = 0; n < num_modes; ++n) {
      qnum3[n] = qsub3[n][jsub] * fcvt_num;
      qnum4[n] = qnum3[n];
      for (int iaer = 0; iaer < num_aerosol_ids; ++iaer) {
        qaer2[iaer][n] = qsub2[iaer][jsub] * fcvt_aer[iaer];
        qaer3[iaer][n] = qsub3[iaer][jsub] * fcvt_aer[iaer];
        qaer4[iaer][n] = qaer3[iaer][n];
      }
      qwtr3[n] = qaerwatsub3[n][jsub] * fcvt_wtr;
      qwtr4[n] = qwtr3[n];
    }
    Real qaercw2[num_aerosol_ids][num_modes] = {};
    Real qaercw3[num_aerosol_ids][num_modes] = {};
    Real qnumcw3[num_modes] = {};
    Real qaercw4[num_aerosol_ids][num_modes] = {};
    Real qnumcw4[num_modes] = {};
    if (iscldy_subarea[jsub]) {
      // only do cloud-borne for cloudy
      for (int n = 0; n < num_modes; ++n) {
        qnumcw3[n] = qqcwsub3[n][jsub] * fcvt_num;
        qnumcw4[n] = qnumcw3[n];
        for (int iaer = 0; iaer < num_aerosol_ids; ++iaer) {
          qaercw2[iaer][n] = qqcwsub2[n][jsub] * fcvt_aer[iaer];
          qaercw3[iaer][n] = qqcwsub3[n][jsub] * fcvt_aer[iaer];
          qaercw4[iaer][n] = qaercw3[iaer][n];
        }
      }
    }

    Real qgas_delaa[num_gas_ids][nqtendaa] = {};
    Real qnum_delaa[num_modes][nqtendaa] = {};
    Real qnumcw_delaa[num_modes][nqqcwtendaa] = {};
    Real qaer_delaa[num_aerosol_ids][num_modes][nqtendaa] = {};
    Real qaercw_delaa[num_aerosol_ids][num_modes][nqqcwtendaa] = {};

    if (iscldy_subarea[jsub]) {
      mam_amicphys_1subarea_cloudy(
          do_cond_sub, do_rename_sub, do_newnuc_sub, do_coag_sub, nstep, deltat,
          jsub, nsubarea, iscldy_subarea[jsub], afracsub[jsub], temp, pmid,
          pdel, zmid, pblh, relhumsub[jsub], dgn_a, dgn_awet, wetdens, qgas1,
          qgas3, qgas4, qgas_delaa, qnum3, qnum4, qnum_delaa, qaer2, qaer3,
          qaer4, qaer_delaa, qwtr3, qwtr4, qnumcw3, qnumcw4, qnumcw_delaa,
          qaercw2, qaercw3, qaercw4, qaercw_delaa);
    } else {
      mam_amicphys_1subarea_clear(
          do_cond_sub, do_rename_sub, do_newnuc_sub, do_coag_sub, nstep, deltat,
          jsub, nsubarea, iscldy_subarea[jsub], afracsub[jsub], temp, pmid,
          pdel, zmid, pblh, relhumsub[jsub], dgn_a, dgn_awet, wetdens, qgas1,
          qgas3, qgas4, qgas_delaa, qnum3, qnum4, qnum_delaa, qaer3, qaer4,
          qaer_delaa, qwtr3, qwtr4);
      // map gas/aer/num arrays (mix-ratio and del=change) back to sub-area
      // arrays

      if (do_map_gas_sub) {
        for (int igas = 0; igas < 2; ++igas) {
          const int l = lmap_gas[igas];
          qsub4[l][jsub] = qgas4[igas] / fcvt_gas[igas];
          for (int i = 0; i < nqtendaa; ++i)
            qsub_tendaa[l][i][jsub] =
                qgas_delaa[igas][i] / (fcvt_gas[igas] * deltat);
        }
      }
      for (int n = 0; n < num_modes; ++n) {
        qsub4[n][jsub] = qnum4[n] / fcvt_num;
        for (int i = 0; i < nqtendaa; ++i)
          qsub_tendaa[n][i][jsub] = qnum_delaa[n][i] / (fcvt_num * deltat);
        for (int iaer = 0; iaer < num_aerosol_ids; ++iaer) {
          qsub4[iaer][jsub] = qaer4[iaer][n] / fcvt_aer[iaer];
          for (int i = 0; i < nqtendaa; ++i)
            qsub_tendaa[iaer][i][jsub] =
                qaer_delaa[iaer][n][i] / (fcvt_aer[iaer] * deltat);
        }
        qaerwatsub4[n][jsub] = qwtr4[n] / fcvt_wtr;

        if (iscldy_subarea[jsub]) {
          qqcwsub4[n][jsub] = qnumcw4[n] / fcvt_num;
          for (int i = 0; i < nqqcwtendaa; ++i)
            qqcwsub_tendaa[n][i][jsub] =
                qnumcw_delaa[n][i] / (fcvt_num * deltat);
          for (int iaer = 0; iaer < num_aerosol_ids; ++iaer) {
            qqcwsub4[iaer][jsub] = qaercw4[iaer][n] / fcvt_aer[iaer];
            for (int i = 0; i < nqqcwtendaa; ++i)
              qqcwsub_tendaa[iaer][i][jsub] =
                  qaercw_delaa[iaer][n][i] / (fcvt_aer[iaer] * deltat);
          }
        }
      }
    }
  }
}

// --------------------------------------------------------------------------------
void subarea_partition_factors(
    const Real
        q_int_cell_avg, // in grid cell mean interstitial aerosol mixing ratio
    const Real
        q_cbn_cell_avg, // in grid cell mean cloud-borne  aerosol mixing ratio
    const Real fcldy,   // in  cloudy fraction of the grid cell
    const Real fclea,   // in clear  fraction of the grid cell
    Real &part_fac_q_int_clea, // out
    Real &part_fac_q_int_cldy) // out
{
  // Calculate mixing ratios of each subarea

  // cloud-borne,  cloudy subarea
  const Real tmp_q_cbn_cldy = q_cbn_cell_avg / fcldy;
  // interstitial, cloudy subarea
  const Real tmp_q_int_cldy =
      haero::max(0.0, ((q_int_cell_avg + q_cbn_cell_avg) - tmp_q_cbn_cldy));
  // interstitial, clear  subarea
  const Real tmp_q_int_clea = (q_int_cell_avg - fcldy * tmp_q_int_cldy) / fclea;

  // Calculate the corresponding paritioning factors for interstitial aerosols
  // using the above-derived subarea mixing ratios plus the constraint that
  // the cloud fraction weighted average of subarea mean need to match grid box
  // mean.

  // *** question ***
  //    use same part_fac_q_int_clea/cldy for everything ?
  //    use one for number and one for all masses (based on total mass) ?
  //    use separate ones for everything ?
  // maybe one for number and one for all masses is best,
  //    because number and mass have different activation fractions
  // *** question ***

  Real tmp_aa = haero::max(1.e-35, tmp_q_int_clea * fclea) /
                haero::max(1.e-35, q_int_cell_avg);
  tmp_aa = haero::max(0.0, haero::min(1.0, tmp_aa));

  part_fac_q_int_clea = tmp_aa / fclea;
  part_fac_q_int_cldy = (1.0 - tmp_aa) / fcldy;
}

// --------------------------------------------------------------------------------

void construct_subareas_1gridcell(
    const Real cld,                        // in
    const Real relhumgcm,                  // in
    const Real q_pregaschem[gas_pcnst],    // in q TMRs before
                                           // gas-phase chemistry
    const Real q_precldchem[gas_pcnst],    // in q TMRs before
                                           // cloud chemistry
    const Real qqcw_precldchem[gas_pcnst], // in  qqcw TMRs before
                                           // cloud chemistry
    const Real q[gas_pcnst],           // in current tracer mixing ratios (TMRs)
                                       // *** MUST BE  #/kmol-air for number
                                       // *** MUST BE mol/mol-air for mass
    const Real qqcw[gas_pcnst],        // in like q but for
                                       // cloud-borner tracers
    int &nsubarea,                     // out
    int &ncldy_subarea,                // out
    int &jclea,                        // out
    int &jcldy,                        // out
    bool iscldy_subarea[maxsubarea],   // out
    Real afracsub[maxsubarea],         // out
    Real relhumsub[maxsubarea],        // out
    Real qsub1[gas_pcnst][maxsubarea], // out interstitial
    Real qsub2[gas_pcnst][maxsubarea], // out interstitial
    Real qsub3[gas_pcnst][maxsubarea], // out interstitial
    Real qqcwsub1[gas_pcnst][maxsubarea], // out cloud-borne
    Real qqcwsub2[gas_pcnst][maxsubarea], // out cloud-borne
    Real qqcwsub3[gas_pcnst][maxsubarea], // outcloud-borne
    Real qaerwatsub3[AeroConfig::num_modes()]
                    [maxsubarea], // out aerosol water mixing ratios (mol/mol)
    Real qaerwat[AeroConfig::num_modes()] // in  aerosol water mixing ratio
                                          // (kg/kg, NOT mol/mol)
) {
  static constexpr int num_modes = AeroConfig::num_modes();
  // cloud chemistry is only on when cld(i,k) >= 1.0e-5_wp
  // it may be that the macrophysics has a higher threshold that this
  const Real fcld_locutoff = 1.0e-5;
  const Real fcld_hicutoff = 0.999;

  // qgcmN and qqcwgcmN (N=1:4) are grid-cell mean tracer mixing ratios (TMRs,
  // mol/mol or #/kmol)
  //    N=1 - before gas-phase chemistry
  //    N=2 - before cloud chemistry
  //    N=3 - incoming values (before gas-aerosol exchange, newnuc, coag)
  //   qgcm1, qgcm2, qgcm3
  //   qqcwgcm2, qqcwgcm3
  //  qaerwatgcm3 ! aerosol water mixing ratios (mol/mol)

  // --------------------------------------------------------------------------------------
  //  Determine the number of sub-areas, their fractional areas, and relative
  //  humidities
  // --------------------------------------------------------------------------------------
  //  if cloud fraction ~= 0, the grid-cell has a single clear  sub-area
  //  (nsubarea = 1) if cloud fraction ~= 1, the grid-cell has a single cloudy
  //  sub-area      (nsubarea = 1) otherwise,              the grid-cell has a
  //  clear and a cloudy sub-area (nsubarea = 2)

  Real zfcldy = 0;
  nsubarea = 0;
  ncldy_subarea = 0;
  jclea = 0;
  jcldy = 0;

  if (cld < fcld_locutoff) {
    nsubarea = 1;
    jclea = 1;
  } else if (cld > fcld_hicutoff) {
    zfcldy = 1.0;
    nsubarea = 1;
    ncldy_subarea = 1;
    jcldy = 1;
  } else {
    zfcldy = cld;
    nsubarea = 2;
    ncldy_subarea = 1;
    jclea = 1;
    jcldy = 2;
  }

  const Real zfclea = 1.0 - zfcldy;
  for (int i = 0; i < maxsubarea; ++i)
    iscldy_subarea[i] = false;
  if (jcldy > 0)
    iscldy_subarea[jcldy - 1] = true;
  for (int i = 0; i < maxsubarea; ++i)
    afracsub[i] = 0.0;
  if (jclea > 0)
    afracsub[jclea - 1] = zfclea;
  if (jcldy > 0)
    afracsub[jcldy - 1] = zfcldy;

  // cldy_rh_sameas_clear is just to match mam_refactor.  Compiler should
  // optimize away.
  const int cldy_rh_sameas_clear = 0;
  if (ncldy_subarea <= 0) {
    for (int i = 0; i < maxsubarea; ++i)
      relhumsub[i] = relhumgcm;
  } else if (cldy_rh_sameas_clear > 0) {
    for (int i = 0; i < maxsubarea; ++i)
      relhumsub[i] = relhumgcm;
  } else {
    if (jcldy > 0) {
      relhumsub[jcldy - 1] = 1.0;
      if (jclea > 0) {
        const Real tmpa =
            (relhumgcm - afracsub[jcldy - 1]) / afracsub[jclea - 1];
        relhumsub[jclea - 1] = haero::max(0.0, haero::min(1.0, tmpa));
      }
    }
  }

  // ----------------------------------------------------------------------------
  //  Copy grid cell mean mixing ratios.
  //  These values, together with cloud fraction and a few assumptions, are used
  //  in the remainder of the subroutine to calculate the sub-area mean mixing
  //  ratios.
  // ----------------------------------------------------------------------------
  //  Interstitial aerosols
  Real qgcm1[gas_pcnst], qgcm2[gas_pcnst], qgcm3[gas_pcnst];
  for (int i = 0; i < gas_pcnst; ++i) {
    qgcm1[i] = haero::max(0.0, q_pregaschem[i]);
    qgcm2[i] = haero::max(0.0, q_precldchem[i]);
    qgcm3[i] = haero::max(0.0, q[i]);
  }

  // Cloud-borne aerosols
  Real qqcwgcm2[gas_pcnst], qqcwgcm3[gas_pcnst];
  for (int i = 0; i < gas_pcnst; ++i) {
    qqcwgcm2[i] = haero::max(0.0, qqcw_precldchem[i]);
    qqcwgcm3[i] = haero::max(0.0, qqcw[i]);
  }

  // aerosol water
  Real qaerwatgcm3[num_modes] = {};
  for (int i = 0; i < num_modes; ++i) {
    qaerwatgcm3[i] = haero::max(0.0, qaerwat[i]);
  }

  // ----------------------------------------------------------------------------
  //  Initialize the subarea mean mixing ratios
  // ----------------------------------------------------------------------------
  {
    const int n = haero::min(maxsubarea, nsubarea + 1);
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < gas_pcnst; ++j) {
        qsub1[j][i] = 0.0;
        qsub2[j][i] = 0.0;
        qsub3[j][i] = 0.0;
        qqcwsub1[j][i] = 0.0;
        qqcwsub2[j][i] = 0.0;
        qqcwsub3[j][i] = 0.0;
      }
      for (int j = 0; j < num_modes; ++j) {
        qaerwatsub3[j][i] = 0.0;
      }
    }
  }

  // *************************************************************************************************
  //  Calculate initial (i.e., before cond/rnam/nnuc/coag) tracer mixing
  //  ratios within the sub-areas
  //   - for all-clear or all-cloudy cases, the sub-area TMRs are equal to the
  //   grid-cell means
  //   - for partly cloudy case, they are different.  This is primarily
  //   because the
  //     interstitial aerosol mixing ratios are assumed lower in the cloudy
  //     sub-area than in the clear sub-area, because much of the aerosol is
  //     activated in the cloudy sub-area.
  // *************************************************************************************************
  //  Category I:  partly cloudy case
  // *************************************************************************************************
  if ((jclea > 0) && (jcldy > 0) && (jclea + jcldy == 3) && (nsubarea == 2)) {

    //  ---------------------------------------------------------------------
    //   Set GAS mixing ratios in sub-areas (for the condensing gases only!!)
    //  ---------------------------------------------------------------------
    for (int lmz = 0; lmz < gas_pcnst; ++lmz) {
      if (lmapcc_all[lmz] == lmapcc_val_gas) {

        // assume gas in both sub-areas before gas-chem and cloud-chem equal
        // grid-cell mean
        for (int i = 0; i < nsubarea; ++i) {
          qsub1[lmz][i] = qgcm1[lmz];
          qsub2[lmz][i] = qgcm2[lmz];
        }
        // assume gas in clear sub-area after cloud-chem equals before
        // cloud-chem value
        qsub3[lmz][jclea - 1] = qsub2[lmz][jclea - 1];
        // gas in cloud sub-area then determined by grid-cell mean and clear
        // values
        qsub3[lmz][jcldy - 1] =
            (qgcm3[lmz] - zfclea * qsub3[lmz][jclea - 1]) / zfcldy;

        // check that this does not produce a negative value
        if (qsub3[lmz][jcldy - 1] < 0.0) {
          qsub3[lmz][jcldy - 1] = 0.0;
          qsub3[lmz][jclea - 1] = qgcm3[lmz] / zfclea;
        }
      }
    }
    // ---------------------------------------------------------------------
    //  Set CLOUD-BORNE AEROSOL mixing ratios in sub-areas.
    //  This is straightforward, as the same partitioning factors (0 or 1/f)
    //  are applied to all mass and number mixing ratios in all modes.
    // ---------------------------------------------------------------------
    // loop thru log-normal modes
    for (int n = 0; n < num_modes; ++n) {
      // number - then mass of individual species - of a mode
      for (int l2 = -1; l2 < num_species_mode(n); ++l2) {
        int lc;
        if (l2 == -1)
          lc = numptr_amode[n];
        else
          lc = lmassptr_amode[l2][n];
        qqcwsub2[lc][jclea - 1] = 0.0;
        qqcwsub2[lc][jcldy - 1] = qqcwgcm2[lc] / zfcldy;
        qqcwsub3[lc][jclea - 1] = 0.0;
        qqcwsub3[lc][jcldy - 1] = qqcwgcm3[lc] / zfcldy;
      }
    }

    // ---------------------------------------------------------------------
    //  Set INTERSTITIAL AEROSOL mixing ratios in sub-areas.
    // ---------------------------------------------------------------------
    for (int n = 0; n < num_modes; ++n) {
      // -------------------------------------
      //  Aerosol number
      // -------------------------------------
      // grid cell mean, interstitial
      Real tmp_q_cellavg_int = qgcm2[numptr_amode[n]];
      // grid cell mean, cloud-borne
      Real tmp_q_cellavg_cbn = qqcwgcm2[numptr_amode[n]];

      Real nmbr_part_fac_clea = 0;
      Real nmbr_part_fac_cldy = 0;
      subarea_partition_factors(tmp_q_cellavg_int, tmp_q_cellavg_cbn, zfcldy,
                                zfclea, nmbr_part_fac_clea, nmbr_part_fac_cldy);

      // Apply the partitioning factors to calculate sub-area mean number
      // mixing ratios

      const int la = numptr_amode[n];

      qsub2[la][jclea - 1] = qgcm2[la] * nmbr_part_fac_clea;
      qsub2[la][jcldy - 1] = qgcm2[la] * nmbr_part_fac_cldy;
      qsub3[la][jclea - 1] = qgcm3[la] * nmbr_part_fac_clea;
      qsub3[la][jcldy - 1] = qgcm3[la] * nmbr_part_fac_cldy;

      //-------------------------------------
      // Aerosol mass
      //-------------------------------------
      // For aerosol mass, we use the total grid cell mean
      // interstitial/cloud-borne mass mixing ratios to come up with the same
      // partitioning for all species in the mode.

      // Compute the total mixing ratios by summing up the individual species

      tmp_q_cellavg_int = 0.0; // grid cell mean, interstitial
      tmp_q_cellavg_cbn = 0.0; // grid cell mean, cloud-borne

      for (int l2 = 0; l2 < num_species_mode(n); ++l2) {
        tmp_q_cellavg_int += qgcm2[lmassptr_amode[l2][n]];
        tmp_q_cellavg_cbn += qqcwgcm2[lmassptr_amode[l2][n]];
      }
      Real mass_part_fac_clea = 0;
      Real mass_part_fac_cldy = 0;
      // Calculate the partitioning factors
      subarea_partition_factors(tmp_q_cellavg_int, tmp_q_cellavg_cbn, zfcldy,
                                zfclea, mass_part_fac_clea, mass_part_fac_cldy);

      // Apply the partitioning factors to calculate sub-area mean mass mixing
      // ratios

      for (int l2 = 0; l2 < num_species_mode(n); ++l2) {
        const int la = lmassptr_amode[l2][n];

        qsub2[la][jclea - 1] = qgcm2[la] * mass_part_fac_clea;
        qsub2[la][jcldy - 1] = qgcm2[la] * mass_part_fac_cldy;
        qsub3[la][jclea - 1] = qgcm3[la] * mass_part_fac_clea;
        qsub3[la][jcldy - 1] = qgcm3[la] * mass_part_fac_cldy;
      }
    }

    // *************************************************************************************************
    //  Category II: all clear, or cld < 1e-5
    //  In this case, zfclea=1 and zfcldy=0
    // *************************************************************************************************
  } else if ((jclea == 1) && (jcldy == 0) && (nsubarea == 1)) {
    //
    // put all the gases and interstitial aerosols in the clear sub-area
    //    and set mix-ratios = 0 in cloudy sub-area
    // for cloud-borne aerosol, do nothing
    //    because the grid-cell-mean cloud-borne aerosol will be left
    //    unchanged (i.e., this routine only changes qqcw when cld >= 1e-5)
    //

    for (int lmz = 0; lmz < gas_pcnst; ++lmz) {
      if (0 < lmapcc_all[lmz]) {
        qsub1[lmz][jclea - 1] = qgcm1[lmz];
        qsub2[lmz][jclea - 1] = qgcm2[lmz];
        qsub3[lmz][jclea - 1] = qgcm3[lmz];
        qqcwsub2[lmz][jclea - 1] = qqcwgcm2[lmz];
        qqcwsub3[lmz][jclea - 1] = qqcwgcm3[lmz];
      }
    }
    // *************************************************************************************************
    //  Category III: all cloudy, or cld > 0.999
    //  in this case, zfcldy= and zfclea=0
    // *************************************************************************************************
  } else if ((jclea == 0) && (jcldy == 1) && (nsubarea == 1)) {
    //
    // put all the gases and interstitial aerosols in the cloudy sub-area
    //    and set mix-ratios = 0 in clear sub-area
    //
    for (int lmz = 0; lmz < gas_pcnst; ++lmz) {
      if (0 < lmapcc_all[lmz]) {
        qsub1[lmz][jcldy - 1] = qgcm1[lmz];
        qsub2[lmz][jcldy - 1] = qgcm2[lmz];
        qsub3[lmz][jcldy - 1] = qgcm3[lmz];
        qqcwsub2[lmz][jcldy - 1] = qqcwgcm2[lmz];
        qqcwsub3[lmz][jcldy - 1] = qqcwgcm3[lmz];
      }
    }
    // *************************************************************************************************
  } else { // this should not happen
    std::cout << "*** modal_aero_amicphys - bad jclea, jcldy, nsubarea" << jclea
              << " " << jcldy << " " << nsubarea << std::endl;
    exit(1);
  }
  // *************************************************************************************************

  // ------------------------------------------------------------------------------------
  //  aerosol water -- how to treat this in sub-areas needs more work/thinking
  //  currently modal_aero_water_uptake calculates qaerwat using
  //     the grid-cell mean interstital-aerosol mix-rats and the clear-area rh
  for (int jsub = 0; jsub < nsubarea; ++jsub)
    for (int i = 0; i < num_modes; ++i)
      qaerwatsub3[i][jsub] = qaerwatgcm3[i];

  // ------------------------------------------------------------------------------------
  if (nsubarea == 1) {
    // the j=1 subarea is used for some diagnostics
    // but is not used in actual calculations
    const int j = 1;
    for (int i = 0; i < gas_pcnst; ++i) {
      qsub1[i][j] = 0.0;
      qsub2[i][j] = 0.0;
      qsub3[i][j] = 0.0;
      qqcwsub2[i][j] = 0.0;
      qqcwsub3[i][j] = 0.0;
    }
  }
}
// ------------------------------------------------------------
// ------------------------------------------------------------
void modal_aero_amicphys_intr(
    const int mdo_gasaerexch, const int mdo_rename, const int mdo_newnuc,
    const int mdo_coag, const int ncol, const int nstep, const Real deltat,
    const Real t, const Real pmid, const Real pdel, const Real zm,
    const Real pblh, const Real qv, const Real cld, Real q[gas_pcnst],
    Real qqcw[gas_pcnst], const Real q_pregaschem[gas_pcnst],
    const Real q_precldchem[gas_pcnst], const Real qqcw_precldchem[gas_pcnst],
    Real q_tendbb[gas_pcnst][nqtendbb], Real qqcw_tendbb[gas_pcnst][nqtendbb],
    Real dgncur_a[AeroConfig::num_modes()],
    Real dgncur_awet[AeroConfig::num_modes()],
    Real wetdens_host[AeroConfig::num_modes()],
    Real qaerwat[AeroConfig::num_modes()]) {

  /*
      ncol                 ! number of atmospheric columns in the chunk
      nstep                ! model time-step number
      nqtendbb             ! dimension for q_tendbb
      nqqcwtendbb          ! dimension f
      deltat               !
      q(ncol,pver,pcnstxx) ! current tracer mixing ratios (TMRs)
                              these values are updated (so out /= in)
                           *** MUST BE  #/kmol-air for number
                           *** MUST BE mol/mol-air for mass
                           *** NOTE ncol dimension
      qqcw(ncol,pver,pcnstxx)
                             like q but for cloud-borner tracers
                            these values are updated
      q_pregaschem(ncol,pver,pcnstxx)    ! q TMRs    before gas-phase
    chemistry q_precldchem(ncol,pver,pcnstxx)    ! q TMRs    before cloud
    chemistry qqcw_precldchem(ncol,pver,pcnstxx) ! qqcw TMRs before cloud
    chemistry q_tendbb(ncol,pver,pcnstxx,nqtendbb)    ! TMR tendencies for
    box-model diagnostic output qqcw_tendbb(ncol,pver,pcnstx t(pcols,pver) !
    temperature at model levels (K) pmid(pcols,pver)     ! pressure at model
    level centers (Pa) pdel(pcols,pver)     ! pressure thickness of levels
    (Pa) zm(pcols,pver)       ! altitude (above ground) at level centers (m)
    pblh(pcols)          ! planetary boundary layer depth (m)
    qv(pcols,pver)       ! specific humidity (kg/kg)
    cld(ncol,pver)       ! cloud fraction (-) *** NOTE ncol dimension
    dgncur_a(pcols,pver,ntot_amode)
    dgncur_awet(pcols,pver,ntot_amode)
                                        ! dry & wet geo. mean dia. (m) of
    number distrib. wetdens_host(pcols,pver,ntot_amode) ! interstitial
    aerosol wet density (kg/m3)

      qaerwat(pcols,pver,ntot_amode    aerosol water mixing ratio (kg/kg,
    NOT mol/mol)

  */

  // !DESCRIPTION:
  // calculates changes to gas and aerosol TMRs (tracer mixing ratios) from
  //    gas-aerosol exchange (condensation/evaporation)
  //    growth from smaller to larger modes (renaming) due to both
  //       condensation and cloud chemistry
  //    new particle nucleation
  //    coagulation
  //    transfer of particles from hydrophobic modes to hydrophilic modes
  //    (aging)
  //       due to condensation and coagulation
  //
  // the incoming mixing ratios (q and qqcw) are updated before output
  //
  // !REVISION HISTORY:
  //   RCE 07.04.13:  Adapted from earlier version of CAM5 modal aerosol
  //   routines
  //                  for these processes
  //

  static constexpr int num_modes = AeroConfig::num_modes();

  // qgcmN and qqcwgcmN (N=1:4) are grid-cell mean tracer mixing ratios
  // (TMRs, mol/mol or #/kmol)
  //    N=1 - before gas-phase chemistry
  //    N=2 - before cloud chemistry
  //    N=3 - incoming values (before gas-aerosol exchange, newnuc, coag)
  //    N=4 - outgoing values (after  gas-aerosol exchange, newnuc, coag)

  // qsubN and qqcwsubN (N=1:4) are TMRs in sub-areas
  //    currently there are just clear and cloudy sub-areas
  //    the N=1:4 have same meanings as for qgcmN

  // q_coltendaa and qqcw_coltendaa are column-integrated tendencies
  //    for different processes, which are output to history
  // the processes are condensation/evaporation (and associated aging),
  //    renaming, coagulation, and nucleation

  const bool do_cond = (mdo_gasaerexch > 0);
  const bool do_rename = (mdo_rename > 0);
  const bool do_newnuc = (mdo_newnuc > 0);
  const bool do_coag = (mdo_coag > 0);

  for (int i = 0; i < gas_pcnst; ++i)
    for (int j = 0; j < nqtendbb; ++j)
      q_tendbb[i][j] = 0.0, qqcw_tendbb[i][j] = 0.0;

  // get saturation mixing ratio
  //     call qsat( t(1:ncol,1:pver), pmid(1:ncol,1:pvnner), &
  //               ev_sat(1:ncol,1:pver), qv_sat(1:ncol,1:pver) )
  const Real epsqs = haero::Constants::weight_ratio_h2o_air;
  // Saturation vapor pressure
  const Real ev_sat = conversions::vapor_saturation_pressure_magnus(t, pmid);
  // Saturation specific humidity
  const Real qv_sat = epsqs * ev_sat / (pmid - (1 - epsqs) * ev_sat);

  const Real relhumgcm = haero::max(0.0, haero::min(1.0, qv / qv_sat));

  // Set up cloudy/clear subareas inside a grid cell
  int nsubarea, ncldy_subarea, jclea, jcldy;
  bool iscldy_subarea[maxsubarea];
  Real afracsub[maxsubarea];
  Real relhumsub[maxsubarea];
  Real qsub1[gas_pcnst][maxsubarea];
  Real qsub2[gas_pcnst][maxsubarea];
  Real qsub3[gas_pcnst][maxsubarea];
  Real qqcwsub1[gas_pcnst][maxsubarea];
  Real qqcwsub2[gas_pcnst][maxsubarea];
  Real qqcwsub3[gas_pcnst][maxsubarea];
  // aerosol water mixing ratios (mol/mol)
  Real qaerwatsub3[AeroConfig::num_modes()][maxsubarea];
  construct_subareas_1gridcell(cld, relhumgcm,                            // in
                               q_pregaschem, q_precldchem,                // in
                               qqcw_precldchem,                           // in
                               q, qqcw,                                   // in
                               nsubarea, ncldy_subarea, jclea, jcldy,     // out
                               iscldy_subarea, afracsub, relhumsub,       // out
                               qsub1, qsub2, qsub3,                       // out
                               qqcwsub1, qqcwsub2, qqcwsub3, qaerwatsub3, // out
                               qaerwat                                    // in
  );

  //  Initialize the "after-amicphys" values
  Real qsub4[gas_pcnst][maxsubarea] = {};
  Real qqcwsub4[gas_pcnst][maxsubarea] = {};
  Real qaerwatsub4[AeroConfig::num_modes()][maxsubarea] = {};

  //
  // start integration
  //
  Real dgn_a[num_modes], dgn_awet[num_modes], wetdens[num_modes];
  for (int n = 0; n < num_modes; ++n) {
    dgn_a[n] = dgncur_a[n];
    dgn_awet[n] = dgncur_awet[n];
    wetdens[n] = haero::max(1000.0, wetdens_host[n]);
  }
  Real qsub_tendaa[gas_pcnst][nqtendaa][maxsubarea] = {};
  Real qqcwsub_tendaa[gas_pcnst][nqqcwtendaa][maxsubarea] = {};
  mam_amicphys_1gridcell(do_cond, do_rename, do_newnuc, do_coag, nstep, deltat,
                         nsubarea, ncldy_subarea, iscldy_subarea, afracsub, t,
                         pmid, pdel, zm, pblh, relhumsub, dgn_a, dgn_awet,
                         wetdens, qsub1, qsub2, qqcwsub2, qsub3, qqcwsub3,
                         qaerwatsub3, qsub4, qqcwsub4, qaerwatsub4, qsub_tendaa,
                         qqcwsub_tendaa);

  //
  // form new grid-mean mix-ratios
  Real qgcm4[gas_pcnst];
  Real qgcm_tendaa[gas_pcnst][nqtendaa];
  Real qaerwatgcm4[num_modes];
  if (nsubarea == 1) {
    for (int i = 0; i < gas_pcnst; ++i)
      qgcm4[i] = qsub4[i][0];
    for (int i = 0; i < gas_pcnst; ++i)
      for (int j = 0; j < nqtendaa; ++j)
        qgcm_tendaa[i][j] = qsub_tendaa[i][j][0];
    for (int i = 0; i < num_modes; ++i)
      qaerwatgcm4[i] = qaerwatsub4[i][0];
  } else {
    for (int i = 0; i < gas_pcnst; ++i)
      qgcm4[i] = 0.0;
    for (int i = 0; i < gas_pcnst; ++i)
      for (int j = 0; j < nqtendaa; ++j)
        qgcm_tendaa[i][j] = 0.0;
    for (int n = 0; n < nsubarea; ++n) {
      for (int i = 0; i < gas_pcnst; ++i)
        qgcm4[i] += qsub4[i][n] * afracsub[n];
      for (int i = 0; i < gas_pcnst; ++i)
        for (int j = 0; j < nqtendaa; ++j)
          qgcm_tendaa[i][j] =
              qgcm_tendaa[i][j] + qsub_tendaa[i][j][n] * afracsub[n];
    }
    for (int i = 0; i < num_modes; ++i)
      // for aerosol water use the clear sub-area value
      qaerwatgcm4[i] = qaerwatsub4[i][jclea - 1];
  }
  Real qqcwgcm4[gas_pcnst];
  Real qqcwgcm_tendaa[gas_pcnst][nqqcwtendaa];
  if (ncldy_subarea <= 0) {
    for (int i = 0; i < gas_pcnst; ++i)
      qqcwgcm4[i] = haero::max(0.0, qqcw[i]);
    for (int i = 0; i < gas_pcnst; ++i)
      for (int j = 0; j < nqqcwtendaa; ++j)
        qqcwgcm_tendaa[i][j] = 0.0;
  } else if (nsubarea == 1) {
    for (int i = 0; i < gas_pcnst; ++i)
      qqcwgcm4[i] = qqcwsub4[i][0];
    for (int i = 0; i < gas_pcnst; ++i)
      for (int j = 0; j < nqqcwtendaa; ++j)
        qqcwgcm_tendaa[i][j] = qqcwsub_tendaa[i][j][0];
  } else {
    for (int i = 0; i < gas_pcnst; ++i)
      qqcwgcm4[i] = 0.0;
    for (int i = 0; i < gas_pcnst; ++i)
      for (int j = 0; j < nqqcwtendaa; ++j)
        qqcwgcm_tendaa[i][j] = 0.0;
    for (int n = 0; n < nsubarea; ++n) {
      if (iscldy_subarea[n]) {
        for (int i = 0; i < gas_pcnst; ++i)
          qqcwgcm4[i] += qqcwsub4[i][n] * afracsub[n];
        for (int i = 0; i < gas_pcnst; ++i)
          for (int j = 0; j < nqqcwtendaa; ++j)
            qqcwgcm_tendaa[i][j] += qqcwsub_tendaa[i][j][n] * afracsub[n];
      }
    }
  }

  for (int lmz = 0; lmz < gas_pcnst; ++lmz) {
    if (lmapcc_all[lmz] > 0) {
      // HW, to ensure non-negative
      q[lmz] = haero::max(qgcm4[lmz], 0.0);
      if (lmapcc_all[lmz] >= lmapcc_val_aer) {
        // HW, to ensure non-negative
        qqcw[lmz] = haero::max(qqcwgcm4[lmz], 0.0);
      }
    }
  }
  for (int i = 0; i < gas_pcnst; ++i) {
    if (iqtend_cond < nqtendbb)
      q_tendbb[i][iqtend_cond] = qgcm_tendaa[i][iqtend_cond];
    if (iqtend_rnam < nqtendbb)
      q_tendbb[i][iqtend_rnam] = qgcm_tendaa[i][iqtend_rnam];
    if (iqtend_nnuc < nqtendbb)
      q_tendbb[i][iqtend_nnuc] = qgcm_tendaa[i][iqtend_nnuc];
    if (iqtend_coag < nqtendbb)
      q_tendbb[i][iqtend_coag] = qgcm_tendaa[i][iqtend_coag];
    if (iqqcwtend_rnam < nqqcwtendbb)
      qqcw_tendbb[i][iqqcwtend_rnam] = qqcwgcm_tendaa[i][iqqcwtend_rnam];
  }
  for (int i = 0; i < num_modes; ++i)
    qaerwat[i] = qaerwatgcm4[i];
}

} // namespace

TEST_CASE("clear", "test_mam4_amicphys") {

  const bool do_cond = true;
  const bool do_rename = true;
  const bool do_newnuc = true;
  const bool do_coag = true;
  const int nstep = 1;
  const Real deltat = 1.0;
  const int jsub = 1;
  const int nsubarea = 2;
  const bool iscldy_subarea = false;
  const Real afracsub = 0.6;
  const Real temp = 273;
  const Real pmid = 100000;
  const Real pdel = 0;
  const Real zmid = 500;
  const Real pblh = 1100;
  const Real relhum = 0.5;

  // clang-format off
  Real dgn_a[AeroConfig::num_modes()] = {
      1.1966637594943708e-007, 2.7656966213478584e-008, 2.1718911947612780e-006, 5.2939487963576523e-008};
  Real dgn_awet[AeroConfig::num_modes()] = {
      1.2730137346646958e-007, 2.9756030032807321e-008, 2.3266870254473761e-006, 5.2963711345420440e-008};
  Real wetdens[AeroConfig::num_modes()] = {
      1311.3148168539929, 1377.8279737773473, 1310.8221954835647, 1490.5543769930989};
  const Real qgas1[AeroConfig::num_gas_ids()] = {0, 0, 1.9196785428799816e-011,
                                                 0, 0, 9.6553333333333350e-011};
  const Real qgas3[AeroConfig::num_gas_ids()] = {0, 0, 1.9196785428799816e-011,
                                                 0, 0, 9.6553333333333350e-011};
  Real qgas4[AeroConfig::num_gas_ids()] = {0, 0, 1.9196785428799816e-011,
                                           0, 0, 9.6553333333333350e-011};
  Real qgas_delaa[AeroConfig::num_gas_ids()][nqtendaa] = {};
  const Real qnum3[AeroConfig::num_modes()] = {
    2253176148.8415728, 22531761488.415726, 2253176.1488415725, 4506352297.6831455};
  Real qnum4[AeroConfig::num_modes()] = {
    2253176148.8415728, 22531761488.415726, 2253176.1488415725, 4506352297.6831455};
  Real qnum_delaa[AeroConfig::num_modes()][nqtendaa] = {
      {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}};
  // clang-format off
  const Real qaer3[AeroConfig::num_aerosol_ids()][AeroConfig::num_modes()] = {
      {2.6484092473027481e-011, 1.9963569502625346e-012, 1.5918312530745280e-010, 0.0000000000000000},
      {3.4543891238118817e-011, 2.6039003387624214e-012, 1.3841779609564249e-010, 0.0000000000000000},
      {8.8280308243424948e-012, 0.0000000000000000,      5.3061041769150928e-011, 1.8930709966834831e-012},
      {0.0000000000000000,      0.0000000000000000,      0.0000000000000000,      9.4653549834174144e-011},
      {6.8037089632648010e-011, 6.8381333387002354e-012, 5.4525090614694067e-010, 0.0000000000000000},
      {0.0000000000000000,      0.0000000000000000,      0.0000000000000000,      0.0000000000000000},
      {0.0000000000000000,      0.0000000000000000,      0.0000000000000000,      0.0000000000000000}};
  Real qaer4[AeroConfig::num_aerosol_ids()][AeroConfig::num_modes()] = {
      {2.6484092473027481e-011, 1.9963569502625346e-012, 1.5918312530745280e-010, 0.0000000000000000},
      {3.4543891238118817e-011, 2.6039003387624214e-012, 1.3841779609564249e-010, 0.0000000000000000},
      {8.8280308243424948e-012, 0.0000000000000000,      5.3061041769150928e-011, 1.8930709966834831e-012},
      {0.0000000000000000,      0.0000000000000000,      0.0000000000000000, 9.4653549834174144e-011},
      {6.8037089632648010e-011, 6.8381333387002354e-012, 5.4525090614694067e-010, 0.0000000000000000},
      {0.0000000000000000,      0.0000000000000000,      0.0000000000000000,      0.0000000000000000},
      {0.0000000000000000,      0.0000000000000000,      0.0000000000000000,      0.0000000000000000}};

  Real qaer_delaa[AeroConfig::num_aerosol_ids()][AeroConfig::num_modes()][nqtendaa] = {};
  const Real qwtr3[AeroConfig::num_modes()] = {
      1.0910189961237504e-010, 9.2545484074841402e-012, 7.3398261784719387e-010, 7.2643814410424918e-014};
  Real qwtr4[AeroConfig::num_modes()] = {
      1.0910189961237504e-010, 9.2545484074841402e-012, 7.3398261784719387e-010, 7.2643814410424918e-014};

  // clang-format on
  // --------------------------------------------------------------------------------
  // --------------------------------------------------------------------------------
  mam_amicphys_1subarea_clear(
      do_cond, do_rename, do_newnuc, do_coag, nstep, deltat, jsub, nsubarea,
      iscldy_subarea, afracsub, temp, pmid, pdel, zmid, pblh, relhum, dgn_a,
      dgn_awet, wetdens, qgas1, qgas3, qgas4, qgas_delaa, qnum3, qnum4,
      qnum_delaa, qaer3, qaer4, qaer_delaa, qwtr3, qwtr4);

  // clang-format off
  const Real check_dgn_a[AeroConfig::num_modes()] = {
      1.1966637594943708e-007, 2.7656966213478584e-008, 2.1718911947612780e-006, 5.2939487963576523e-008};
  const Real check_dgn_awet[AeroConfig::num_modes()] = {
      1.2730137346646958e-007, 2.9756030032807321e-008, 2.3266870254473761e-006, 5.2963711345420440e-008};
  const Real check_wetdens[AeroConfig::num_modes()] = {
      1311.3148168539929, 1377.8279737773473, 1310.8221954835647, 1490.5543769930989};
  const Real check_qgas4[AeroConfig::num_gas_ids()] = {0, 0, 1.9012167647987512e-011,
                                                       0, 0, 9.6509455923714994e-011};
  const Real check_qgas_delaa[AeroConfig::num_gas_ids()][nqtendaa] = {
      {}, {},
      {-1.0806949256924317e-014, 0.0000000000000000, -1.7431083155538128e-013, 0.0000000000000000, -1.0806949256924317e-014 },
      {}, {},
      {-4.3877409618355718e-014, 0.0000000000000000,  0.0000000000000000,      0.0000000000000000, -4.3877409618355718e-014 }};
  const Real check_qnum4[AeroConfig::num_modes()] = {
      2276787216.2584395l, 31806406005.471962, 2253176.1488415725, 4484126983.6989059};
  const Real check_qnum_delaa[AeroConfig::num_modes()][nqtendaa] = {
      { 22217892.476436529, 1388141.8426356316, 0.0000000000000000,  5033.0977945179447, 0.0000000000000000},
      { 0.0000000000000000,-1388141.8426361084, 9276121037.9203491, -88379.021476745605, 0.0000000000000000},
      { 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,  0.0000000000000000, 0.0000000000000000},
      {-22217892.476436529, 0.0000000000000000, 0.0000000000000000, -7421.5078029484257, 0.0000000000000000}};
  const Real
      check_qaer4[AeroConfig::num_aerosol_ids()][AeroConfig::num_modes()] = {
      {2.6540116468902412e-011 , 2.0083963042364056e-012 , 1.5932627272135888e-010 , 0.0000000000000000      },
      {3.4554314745585923e-011 , 2.7782968196239490e-012 , 1.3841809388812617e-010 , 0.0000000000000000      },
      {8.8452468223765214e-012 , 0.0000000000000000      , 5.3108354531395086e-011 , 1.8854146477732043e-012 },
      {4.6680063022077775e-013 , 0.0000000000000000      , 0.0000000000000000      , 9.4186749203953365e-011 },
      {6.8044994737478904e-011 , 6.8302282338693398e-012 , 5.4525090614694067e-010 , 0.0000000000000000      },
      {0.0000000000000000      , 0.0000000000000000      , 0.0000000000000000      , 0.0000000000000000      },
      {0.0000000000000000      , 0.0000000000000000      , 0.0000000000000000      , 0.0000000000000000      }};
  const Real check_qaer_delaa
      [AeroConfig::num_aerosol_ids()][AeroConfig::num_modes()][nqtendaa] = {
      {{ 3.0084554734817190e-014,  2.3231420098151164e-015, 0.0000000000000000,       1.3166734628086775e-018, 2.2461700794338613e-014},
       { 1.2583727709935030e-014, -2.3231420098153086e-015, 0.0000000000000000,      -1.3166734628086798e-018, 1.2583727709935030e-014},
       { 1.2091271736360237e-015,  0.0000000000000000,      0.0000000000000000,       0.0000000000000000,      1.2091271736360237e-015},
       { 0.0000000000000000,       0.0000000000000000,      0.0000000000000000,       0.0000000000000000,      7.6228539404785740e-015}},


      {{ 7.4096164316262002e-015,  3.0120696271787838e-015, 0.0000000000000000,       1.8214080493613943e-018, 5.5320287095013882e-015},
       { 3.0995403416210632e-015, -3.0120696271787893e-015, 1.7431083155538087e-013, -1.8214080475994735e-018, 3.0995403416210632e-015},
       { 2.9779248367861854e-016,  0.0000000000000000,      0.0000000000000000,       0.0000000000000000,      2.9779248367861854e-016},
       { 0.0000000000000000,       0.0000000000000000,      0.0000000000000000,       0.0000000000000000,      1.8775877221248125e-015}},

      {{ 9.3418273099770602e-015,  0.0000000000000000,      0.0000000000000000,       2.5099056774641765e-018, 0.0000000000000000},
       { 0.0000000000000000,       0.0000000000000000,      0.0000000000000000,       0.0000000000000000,      0.0000000000000000},
       { 0.0000000000000000,       0.0000000000000000,      0.0000000000000000,       0.0000000000000000,      0.0000000000000000},
       {-9.3418273099770602e-015,  0.0000000000000000,      0.0000000000000000,      -2.5099056774641765e-018, 0.0000000000000000}},

      {{ 4.6667524673717908e-013,  0.0000000000000000,      0.0000000000000000,       1.2538348359869531e-016, 0.0000000000000000},
       { 0.0000000000000000,       0.0000000000000000,      0.0000000000000000,       0.0000000000000000,      0.0000000000000000},
       { 0.0000000000000000,       0.0000000000000000,      0.0000000000000000,       0.0000000000000000,      0.0000000000000000},
       {-4.6667524673717908e-013,  0.0000000000000000,      0.0000000000000000,      -1.2538348359869531e-016, 0.0000000000000000}},

      {{ 0.0000000000000000,       7.9006270392206911e-015, 0.0000000000000000,       4.4777910393789187e-018, 0.0000000000000000},
       { 0.0000000000000000,      -7.9006270392206911e-015, 0.0000000000000000,      -4.4777910424481607e-018, 0.0000000000000000},
       { 0.0000000000000000,       0.0000000000000000,      0.0000000000000000,       0.0000000000000000,       0.0000000000000000},
       { 0.0000000000000000,       0.0000000000000000,      0.0000000000000000,       0.0000000000000000,       0.0000000000000000}},

      {{ 0.0000000000000000,       0.0000000000000000,      0.0000000000000000,       0.0000000000000000,       0.0000000000000000},
       { 0.0000000000000000,       0.0000000000000000,      0.0000000000000000,       0.0000000000000000,       0.0000000000000000},
       { 0.0000000000000000,       0.0000000000000000,      0.0000000000000000,       0.0000000000000000,       0.0000000000000000},
       { 0.0000000000000000,       0.0000000000000000,      0.0000000000000000,       0.0000000000000000,       0.0000000000000000}},

      {{ 0.0000000000000000,       0.0000000000000000,      0.0000000000000000,       0.0000000000000000,       0.0000000000000000},
       { 0.0000000000000000,       0.0000000000000000,      0.0000000000000000,       0.0000000000000000,       0.0000000000000000},
       { 0.0000000000000000,       0.0000000000000000,      0.0000000000000000,       0.0000000000000000,       0.0000000000000000},
       { 0.0000000000000000,       0.0000000000000000,      0.0000000000000000,       0.0000000000000000,       0.0000000000000000}}};

  const Real check_qwtr4[AeroConfig::num_modes()] = {
      1.0910189961237504e-010, 9.2545484074841402e-012, 7.3398261784719387e-010, 7.2643814410424918e-014};

  // clang-format on
  for (int i = 0; i < AeroConfig::num_modes(); ++i)
    CHECK(!isnan(dgn_a[i]));
  for (int i = 0; i < AeroConfig::num_modes(); ++i)
    CHECK(!isnan(dgn_awet[i]));
  for (int i = 0; i < AeroConfig::num_modes(); ++i)
    CHECK(!isnan(wetdens[i]));
  for (int i = 0; i < AeroConfig::num_gas_ids(); ++i)
    CHECK(!isnan(qgas4[i]));
  for (int i = 0; i < AeroConfig::num_gas_ids(); ++i)
    for (int j = 0; j < nqtendaa; ++j)
      CHECK(!isnan(qgas_delaa[i][j]));
  for (int i = 0; i < AeroConfig::num_modes(); ++i)
    CHECK(!isnan(qnum4[i]));
  for (int i = 0; i < AeroConfig::num_modes(); ++i)
    for (int j = 0; j < nqtendaa; ++j)
      CHECK(!isnan(qnum_delaa[i][j]));
  for (int i = 0; i < AeroConfig::num_aerosol_ids(); ++i)
    for (int j = 0; j < AeroConfig::num_modes(); ++j)
      CHECK(!isnan(qaer4[i][j]));
  for (int i = 0; i < AeroConfig::num_aerosol_ids(); ++i)
    for (int j = 0; j < AeroConfig::num_modes(); ++j)
      for (int k = 0; k < nqtendaa; ++k)
        CHECK(!isnan(qaer_delaa[i][j][k]));
  for (int i = 0; i < AeroConfig::num_modes(); ++i)
    CHECK(!isnan(qwtr4[i]));
  for (int i = 0; i < AeroConfig::num_modes(); ++i) {
    if (!(dgn_a[i] == Approx(check_dgn_a[i])))
      std::cout << "dgn_a[i] != Approx(check_dgn_a[i])): "
                << std::setprecision(14) << dgn_a[i] << " != " << check_dgn_a[i]
                << std::endl;
    REQUIRE(dgn_a[i] == Approx(check_dgn_a[i]));
  }
  for (int i = 0; i < AeroConfig::num_modes(); ++i) {
    if (!(dgn_awet[i] == Approx(check_dgn_awet[i])))
      std::cout << "dgn_awet[i] != Approx(check_dgn_awet[i])): "
                << std::setprecision(14) << dgn_awet[i]
                << " != " << check_dgn_awet[i] << std::endl;
    REQUIRE(dgn_awet[i] == Approx(check_dgn_awet[i]));
  }
  for (int i = 0; i < AeroConfig::num_modes(); ++i) {
    if (!(wetdens[i] == Approx(check_wetdens[i])))
      std::cout << "wetdens[" << i << "] != Approx(check_wetdens[" << i
                << "])): " << std::setprecision(14) << wetdens[i]
                << " != " << check_wetdens[i] << std::endl;
    REQUIRE(wetdens[i] == Approx(check_wetdens[i]));
  }
  for (int i = 0; i < AeroConfig::num_gas_ids(); ++i) {
    const double epsilon = 0.0001;
    const double scale = std::abs(check_qgas4[i]);
    if (!(qgas4[i] == Approx(check_qgas4[i]).scale(scale).epsilon(epsilon))) {
      std::cout << __FILE__ << ":" << __LINE__ << "qgas4[" << i
                << "] != Approx(check_qgas4[" << i
                << "])): " << std::setprecision(14) << qgas4[i]
                << " != " << check_qgas4[i] << " with scale:" << scale
                << " and epsilon:" << epsilon << std::endl;
      // std::cout << "i = " << i << "\n";
      // std::cout << "qgas4[i] = " << qgas4[i] << "\n";
      // std::cout << "check_qgas4[i] = " << check_qgas4[i] << "\n";
    }
    REQUIRE(qgas4[i] == Approx(check_qgas4[i]).scale(scale).epsilon(epsilon));
  }
  for (int i = 0; i < AeroConfig::num_gas_ids(); ++i) {
    for (int j = 0; j < nqtendaa; ++j) {
      const double epsilon = std::is_same_v<Real, double> ? 0.0001 : 0.001;
      const double scale = std::abs(check_qgas_delaa[i][j]);
      if (!(qgas_delaa[i][j] ==
            Approx(check_qgas_delaa[i][j]).scale(scale).epsilon(epsilon)))
        std::cout << "qgas_delaa[" << i << "][" << j
                  << "] != Approx(check_qgas_delaa[" << i << "][" << j
                  << "])): " << std::setprecision(14) << qgas_delaa[i][j]
                  << " != " << check_qgas_delaa[i][j] << " with scale:" << scale
                  << " and epsilon:" << epsilon << std::endl;
      REQUIRE(qgas_delaa[i][j] ==
              Approx(check_qgas_delaa[i][j]).scale(scale).epsilon(epsilon));
    }
  }
  for (int i = 0; i < AeroConfig::num_modes(); ++i) {
    const double epsilon = 0.004;
    const double scale = std::abs(check_qnum4[i]);
    const bool check =
        qnum4[i] == Approx(check_qnum4[i]).scale(scale).epsilon(epsilon);
    if (!check)
      std::cout << "qnum4[" << i << "] != Approx(check_qnum4[" << i
                << "])): " << std::setprecision(14) << qnum4[i]
                << " != " << check_qnum4[i] << " with scale:" << scale
                << " and epsilon:" << epsilon << std::endl;
    REQUIRE(qnum4[i] == Approx(check_qnum4[i]).scale(scale).epsilon(epsilon));
  }
  if (std::is_same_v<Real, double>)
    for (int i = 0; i < AeroConfig::num_modes(); ++i) {
      for (int j = 0; j < nqtendaa; ++j) {
        const double epsilon = 0.0005;
        const double scale = std::abs(check_qnum_delaa[i][j]);
        const bool check =
            qnum_delaa[i][j] ==
            Approx(check_qnum_delaa[i][j]).scale(scale).epsilon(epsilon);
        if (!check)
          std::cout << "qnum_delaa[" << i << "][" << j
                    << "] != Approx(check_qnum_delaa[" << i << "][" << j
                    << "])): " << std::setprecision(14) << qnum_delaa[i][j]
                    << " != " << check_qnum_delaa[i][j]
                    << " with scale:" << scale << " and epsilon:" << epsilon
                    << std::endl;
        REQUIRE(qnum_delaa[i][j] ==
                Approx(check_qnum_delaa[i][j]).scale(scale).epsilon(epsilon));
      }
    }
  for (int i = 0; i < AeroConfig::num_aerosol_ids(); ++i) {
    for (int j = 0; j < AeroConfig::num_modes(); ++j) {
      const double epsilon = 0.0005;
      const double scale = std::abs(check_qaer4[i][j]);
      if (check_qaer4[i][j]) {
        if (!(qaer4[i][j] ==
              Approx(check_qaer4[i][j]).scale(scale).epsilon(epsilon)))
          std::cout << "qaer4[" << i << "][" << j << "] != Approx(check_qaer4["
                    << i << "][" << j << "])): " << std::setprecision(14)
                    << qaer4[i][j] << " != " << check_qaer4[i][j]
                    << " with scale:" << scale << " and epsilon:" << epsilon
                    << std::endl;
        REQUIRE(qaer4[i][j] ==
                Approx(check_qaer4[i][j]).scale(scale).epsilon(epsilon));
      }
    }
  }
  if (std::is_same_v<Real, double>)
    for (int i = 0; i < AeroConfig::num_aerosol_ids(); ++i) {
      for (int j = 0; j < AeroConfig::num_modes(); ++j) {
        for (int k = 0; k < nqtendaa; ++k) {
          const double epsilon = 0.0005;
          const double scale = std::abs(check_qaer_delaa[i][j][k]);
          if (!(qaer_delaa[i][j][k] == Approx(check_qaer_delaa[i][j][k])
                                           .scale(scale)
                                           .epsilon(epsilon)))
            std::cout << "qaer_delaa[" << i << "][" << j << "][" << k
                      << "] != Approx(check_qaer_delaa[" << i << "][" << j
                      << "][" << k << "])): " << std::setprecision(14)
                      << qaer_delaa[i][j][k]
                      << " != " << check_qaer_delaa[i][j][k]
                      << " with scale:" << scale << " and epsilon:" << epsilon
                      << std::endl;
          REQUIRE(
              qaer_delaa[i][j][k] ==
              Approx(check_qaer_delaa[i][j][k]).scale(scale).epsilon(epsilon));
        }
      }
    }
  for (int i = 0; i < AeroConfig::num_modes(); ++i) {
    if (!(qwtr4[i] == Approx(check_qwtr4[i])))
      std::cout << "qwtr4[" << i << "] != Approx(check_qwtr4[" << i
                << "])): " << std::setprecision(14) << qwtr4[i]
                << " != " << check_qwtr4[i] << std::endl;
    REQUIRE(qwtr4[i] == Approx(check_qwtr4[i]));
  }
}

TEST_CASE("cloudy", "test_mam4_amicphys") {
  static constexpr int num_gas_ids = AeroConfig::num_gas_ids();
  static constexpr int num_modes = AeroConfig::num_modes();
  static constexpr int num_aerosol_ids = AeroConfig::num_aerosol_ids();
  const bool do_cond_sub = true;
  const bool do_rename_sub = true;
  const bool do_newnuc_sub = true;
  const bool do_coag_sub = true;
  const int nstep = 1;
  const Real deltat = 1.0000000000000000;
  const int jsub = 2;
  const int nsubarea = 2;
  const bool iscldy_subarea = true;

  const Real afracsub = 0.40000000000000002;
  const Real temp = 273.00000000000000;
  const Real pmid = 100000.00000000000;
  const Real pdel = 0.00000000000;
  const Real zmid = 500.00000000000000;
  const Real pblh = 1100.0000000000000;
  const Real relhumsub = 1.0000000000000000;

  Real dgn_a[num_modes] = {1.1966637594943708e-007, 2.7656966213478584e-008,
                           2.1718911947612780e-006, 5.2939487963576523e-008};
  Real dgn_awet[num_modes] = {1.2730137346646958e-007, 2.9756030032807321e-008,
                              2.3266870254473761e-006, 5.2963711345420440e-008};
  Real wetdens[num_modes] = {1311.3148168539929, 1377.8279737773473,
                             1310.8221954835647, 1490.5543769930989};
  const Real qgas1[num_gas_ids] = {};
  const Real qgas3[num_gas_ids] = {};
  Real qgas4[num_gas_ids] = {};
  // clang-format off
  Real qgas_delaa[num_gas_ids][nqtendaa] = {
      {}, {},
      {-1.0806949256924317e-014, 0.0000000000000000, -1.7431083155538128e-013, 0.0000000000000000, -1.0806949256924317e-014},
      {}, {},
      {-4.3877409618355718e-014, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, -4.3877409618355718e-014}};
  const Real qnum3[num_modes] = {2253176148.8415728, 22531761488.415726, 2253176.1488415725, 4506352297.6831455};
  Real qnum4[num_modes] = {2253176148.8415728, 22531761488.415726, 2253176.1488415725, 4506352297.6831455};
  Real qnum_delaa[num_modes][nqtendaa] = {
      {22237702.627951536, 1387797.7896466255, 0.0000000000000000, 5036.2946107021908},
      {0.0000000000000000, -1387797.7896461487, 9276121037.9203491, -88379.019798278809},
      {0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000},
      {-22237702.627951536, 0.0000000000000000, 0.0000000000000000, -7424.7043602100948}};
  const Real qaer2[num_aerosol_ids][num_modes] = {
      {2.6484092473027475e-011, 1.9963569502625334e-012, 1.5918312530745288e-010, 0.0000000000000000},
      {3.4543891238118804e-011, 2.6039003387624206e-012, 1.3841779609564254e-010, 0.0000000000000000},
      {8.8280308243424915e-012, 0.0000000000000000, 5.3061041769150948e-011, 1.8930709966834831e-012},
      {0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 9.4653549834174144e-011},
      {6.8037089632647984e-011, 6.8381333387002322e-012, 5.4525090614694088e-010, 0.0000000000000000},
      {},
      {}};
  const Real qaer3[num_aerosol_ids][num_modes] = {
      {2.6484092473027475e-011, 1.9963569502625334e-012, 1.5918312530745288e-010, 0.0000000000000000},
      {3.4543891238118804e-011, 2.6039003387624206e-012, 1.3841779609564254e-010, 0.0000000000000000},
      {8.8280308243424915e-012, 0.0000000000000000, 5.3061041769150948e-011, 1.8930709966834831e-012},
      {0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 9.4653549834174144e-011},
      {6.8037089632647984e-011, 6.8381333387002322e-012, 5.4525090614694088e-010, 0.0000000000000000},
      {},
      {}};
  Real qaer4[num_aerosol_ids][num_modes] = {
      {2.6484092473027475e-011, 1.9963569502625334e-012, 1.5918312530745288e-010, 0.0000000000000000},
      {3.4543891238118804e-011, 2.6039003387624206e-012, 1.3841779609564254e-010, 0.0000000000000000},
      {8.8280308243424915e-012, 0.0000000000000000, 5.3061041769150948e-011, 1.8930709966834831e-012},
      {0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 9.4653549834174144e-011},
      {6.8037089632647984e-011, 6.8381333387002322e-012, 5.4525090614694088e-010, 0.0000000000000000},
      {},
      {}};
  Real qaer_delaa[num_aerosol_ids][num_modes][nqtendaa] = {
      {{3.0084554734817190e-014, 2.3212443801711164e-015, 0.0000000000000000, 1.3155076303831375e-018, 2.2461700794338613e-014},
       {1.2583727709935030e-014, -2.3212443801703086e-015, 0.0000000000000000, -1.3155076286571098e-018, 1.2583727709935030e-014},
       {1.2091271736360237e-015, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 1.2091271736360237e-015},
       {0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 7.6228539404785740e-015}},
      {{7.4096164316262002e-015, 3.0122760143315438e-015, 0.0000000000000000, 1.8214080493613943e-018, 5.5320287095013882e-015},
       {3.0995403416210632e-015, -3.0122760143331593e-015, 1.7431083155538087e-013, -1.8214080475994735e-018, 3.0995403416210632e-015},
       {2.9779248367861854e-016, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 2.9779248367861854e-016},
       {0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 1.8775877221248125e-015}},
      {{9.3418269728088690e-015, 0.0000000000000000, 0.0000000000000000, 2.5090125486423108e-018, 0.0000000000000000},
       {0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000},
       {0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000},
       {-9.3418269728088690e-015, 0.0000000000000000, 0.0000000000000000, -2.5090125490462076e-018, 0.0000000000000000}},
      {{4.6709134864044329e-013, 0.0000000000000000, 0.0000000000000000, 1.2545062745841079e-016, 0.0000000000000000},
       {0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000},
       {0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000},
       {-4.6709134864044329e-013, 0.0000000000000000, 0.0000000000000000, -1.2545062745473374e-016, 0.0000000000000000}},
      {{0.0000000000000000, 7.9011683905581815e-015, 0.0000000000000000, 4.4777910393789187e-018, 0.0000000000000000},
       {0.0000000000000000, -7.9011683905638360e-015, 0.0000000000000000, -4.4777910424481607e-018, 0.0000000000000000},
       {},
       {}},
      {{}, {}, {}, {}},
      {{}, {}, {}, {}}};
  const Real qwtr3[num_modes] = {
      1.0910189961237504e-010, 9.2545484074841402e-012, 7.3398261784719387e-010, 7.2643814410424918e-014};
  Real qwtr4[num_modes] = {1.0910189961237504e-010, 9.2545484074841402e-012, 7.3398261784719387e-010, 7.2643814410424918e-014};
  const Real qnumcw3[num_modes] = {};
  Real qnumcw4[num_modes] = {};
  Real qnumcw_delaa[num_modes][nqqcwtendaa] = {};
  const Real qaercw2[num_aerosol_ids][num_modes] = {};
  const Real qaercw3[num_aerosol_ids][num_modes] = {};
  Real qaercw4[num_aerosol_ids][num_modes] = {};
  Real qaercw_delaa[num_aerosol_ids][num_modes][nqqcwtendaa] = {};
  // clang-format on
  //--------------------------------------------------------------------------------
  //--------------------------------------------------------------------------------
  mam_amicphys_1subarea_cloudy(
      do_cond_sub, do_rename_sub, do_newnuc_sub, do_coag_sub, nstep, deltat,
      jsub, nsubarea, iscldy_subarea, afracsub, temp, pmid, pdel, zmid, pblh,
      relhumsub, dgn_a, dgn_awet, wetdens, qgas1, qgas3, qgas4, qgas_delaa,
      qnum3, qnum4, qnum_delaa, qaer2, qaer3, qaer4, qaer_delaa, qwtr3, qwtr4,
      qnumcw3, qnumcw4, qnumcw_delaa, qaercw2, qaercw3, qaercw4, qaercw_delaa);

  // clang-format off
  const Real check_dgn_a[num_modes] = {
      1.1966637594943708e-007, 2.7656966213478584e-008, 2.1718911947612780e-006, 5.2939487963576523e-008};
  const Real check_dgn_awet[num_modes] = {
      1.2730137346646958e-007, 2.9756030032807321e-008, 2.3266870254473761e-006, 5.2963711345420440e-008};
  const Real check_wetdens[num_modes] = {
      1311.3148168539929, 1377.8279737773473, 1310.8221954835647, 1490.5543769930989};
  const Real check_qgas4[num_gas_ids] = {0.0, 0.0, 0.0, 0.0, 0.0, 1.1708228956931740e-016};
  const Real check_qgas_delaa[num_gas_ids][nqtendaa] = {
      {}, {}, {}, {}, {},
      {1.1708228956931740e-016,  0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 1.1708228956931740e-016}};
  const Real check_qnum4[num_modes] = {
      2253176148.8415728, 22531761488.415726, 2253176.1488415725, 4506352297.6831455};
  const Real check_qnum_delaa[num_modes][nqtendaa] = {
      {7.7571933637159115e-003,  0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000},
      {0.0000000000000000     ,  0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000},
      {0.0000000000000000     ,  0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000},
      {-7.7571933637159115e-003,  0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000}};

  const Real check_qaer4[num_aerosol_ids][num_modes] = {
    {2.6484020766304678e-011,  1.9963154347230245e-012,   1.5918312144742561e-010,   0.0000000000000000      },
    {3.4543891238118804e-011,  2.6039003387624206e-012,   1.3841779609564254e-010,   0.0000000000000000      },
    {8.8280308243457502e-012,  0.0000000000000000     ,   5.3061041769150948e-011,   1.8930709966802244e-012 },
    {1.6293574938721667e-022,  0.0000000000000000     ,   0.0000000000000000     ,   9.4653549834011203e-011 },
    {6.8037089632647984e-011,  6.8381333387002322e-012,   5.4525090614694088e-010,   0.0000000000000000      },
    {0.0000000000000000     ,  0.0000000000000000     ,   0.0000000000000000     ,   0.0000000000000000      },
    {0.0000000000000000     ,  0.0000000000000000     ,   0.0000000000000000     ,   0.0000000000000000      }
  };
  const Real check_qaer_delaa[num_aerosol_ids][num_modes][nqtendaa] = {
 {{ -7.1706722797035180e-017, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, -7.1706726483491915e-017},
  { -4.1515539508818071e-017, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, -4.1515539508818071e-017},
  { -3.8600272649380107e-018, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, -3.8600272649380107e-018},
  {  0.0000000000000000     , 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,  3.6864567361695183e-024}},
 {{  0.0000000000000000     , 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,  0.0000000000000000},
  {  0.0000000000000000     , 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,  0.0000000000000000},
  {  0.0000000000000000     , 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,  0.0000000000000000},
  {  0.0000000000000000     , 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,  0.0000000000000000}},
 {{  3.2587149877443334e-024, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,  0.0000000000000000},
  {  0.0000000000000000     , 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,  0.0000000000000000},
  {  0.0000000000000000     , 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,  0.0000000000000000},
  { -3.2587149877443334e-024, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,  0.0000000000000000}},
 {{  1.6293574938721667e-022, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,  0.0000000000000000},
  {  0.0000000000000000     , 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,  0.0000000000000000},
  {  0.0000000000000000     , 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,  0.0000000000000000},
  { -1.6293574938721667e-022, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,  0.0000000000000000}},
 {{  0.0000000000000000     , 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,  0.0000000000000000},
  {  0.0000000000000000     , 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,  0.0000000000000000},
  {  0.0000000000000000     , 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,  0.0000000000000000},
  {  0.0000000000000000     , 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,  0.0000000000000000}},
 {{  0.0000000000000000     , 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,  0.0000000000000000},
  {  0.0000000000000000     , 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,  0.0000000000000000},
  {  0.0000000000000000     , 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,  0.0000000000000000},
  {  0.0000000000000000     , 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,  0.0000000000000000}},
 {{  0.0000000000000000     , 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,  0.0000000000000000},
  {  0.0000000000000000     , 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,  0.0000000000000000},
  {  0.0000000000000000     , 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,  0.0000000000000000},
  {  0.0000000000000000     , 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,  0.0000000000000000}}};

  const Real check_qwtr4[num_modes] = {
      1.0910189961237504e-010, 9.2545484074841402e-012, 7.3398261784719387e-010, 7.2643814410424918e-014};
  // clang-format on
  for (int i = 0; i < num_modes; ++i) {
    if (!(dgn_a[i] == Approx(check_dgn_a[i])))
      std::cout << "dgn_a[i] != Approx(check_dgn_a[i])): "
                << std::setprecision(14) << dgn_a[i] << " != " << check_dgn_a[i]
                << std::endl;
    REQUIRE(dgn_a[i] == Approx(check_dgn_a[i]));
  }
  for (int i = 0; i < num_modes; ++i) {
    if (!(dgn_awet[i] == Approx(check_dgn_awet[i])))
      std::cout << "dgn_awet[i] != Approx(check_dgn_awet[i])): "
                << std::setprecision(14) << dgn_awet[i]
                << " != " << check_dgn_awet[i] << std::endl;
    REQUIRE(dgn_awet[i] == Approx(check_dgn_awet[i]));
  }
  for (int i = 0; i < num_modes; ++i) {
    if (!(wetdens[i] == Approx(check_wetdens[i])))
      std::cout << "wetdens[i] != Approx(check_wetdens[i])): "
                << std::setprecision(14) << wetdens[i]
                << " != " << check_wetdens[i] << std::endl;
    REQUIRE(wetdens[i] == Approx(check_wetdens[i]));
  }
  for (int i = 0; i < num_gas_ids; ++i) {
#ifdef HAERO_DOUBLE_PRECISION
    const double epsilon = 0.0001;
#else
    const double epsilon = 0.1;
#endif
    const double scale = std::abs(check_qgas4[i]);
    if (!(qgas4[i] == Approx(check_qgas4[i]).epsilon(epsilon).scale(scale)))
      std::cout << "qgas4[" << i << "] != Approx(check_qgas4[" << i
                << "])).epsilon(" << epsilon << ").scale(" << scale
                << "): " << std::setprecision(14) << qgas4[i]
                << " != " << check_qgas4[i] << std::endl;
    REQUIRE(qgas4[i] == Approx(check_qgas4[i]).epsilon(epsilon).scale(scale));
  }
  for (int i = 0; i < num_gas_ids; ++i) {
    for (int j = 0; j < nqtendaa; ++j) {
#ifdef HAERO_DOUBLE_PRECISION
      const double epsilon = 0.0001;
#else
      const double epsilon = 0.1;
#endif
      const double scale = std::abs(qgas_delaa[i][j]);
      if (!(qgas_delaa[i][j] ==
            Approx(check_qgas_delaa[i][j]).epsilon(epsilon).scale(scale)))
        std::cout << "qgas_delaa[" << i << "][" << j
                  << "] != Approx(check_qgas_delaa[" << i << "][" << j
                  << "]).epsilon(" << epsilon << ").scale(" << scale
                  << "): " << std::setprecision(14) << qgas_delaa[i][j]
                  << " != " << check_qgas_delaa[i][j] << std::endl;
      REQUIRE(qgas_delaa[i][j] ==
              Approx(check_qgas_delaa[i][j]).epsilon(epsilon).scale(scale));
    }
  }
  for (int i = 0; i < num_modes; ++i) {
    if (!(qnum4[i] == Approx(check_qnum4[i])))
      std::cout << "qnum4[i] != Approx(check_qnum4[i])): "
                << std::setprecision(14) << qnum4[i] << " != " << check_qnum4[i]
                << std::endl;
    REQUIRE(qnum4[i] == Approx(check_qnum4[i]));
  }
  for (int i = 0; i < num_modes; ++i) {
    for (int j = 0; j < nqtendaa; ++j) {
#ifdef HAERO_DOUBLE_PRECISION
      const double epsilon = 0.001;
#else
      const double epsilon = 0.5;
#endif
      const double scale = std::abs(qnum_delaa[i][j]);
      if (!(qnum_delaa[i][j] ==
            Approx(check_qnum_delaa[i][j]).epsilon(epsilon).scale(scale)))
        std::cout << "qnum_delaa[" << i << "][" << j
                  << "] != Approx(check_qnum_delaa[" << i << "][" << j
                  << "]epsilon(" << epsilon << ").scale(" << scale
                  << "): " << std::setprecision(14) << qnum_delaa[i][j]
                  << " != " << check_qnum_delaa[i][j] << std::endl;
      REQUIRE(qnum_delaa[i][j] ==
              Approx(check_qnum_delaa[i][j]).epsilon(epsilon).scale(scale));
    }
  }
  for (int i = 0; i < num_aerosol_ids; ++i) {
    for (int j = 0; j < num_modes; ++j) {
#ifdef HAERO_DOUBLE_PRECISION
      const double epsilon = 0.001;
#else
      const double epsilon = 0.1;
#endif
      const double scale = std::abs(qaer4[i][j]);
      if (!(qaer4[i][j] ==
            Approx(check_qaer4[i][j]).epsilon(epsilon).scale(scale)))
        std::cout << "qaer4[" << i << "][" << j << "] != Approx(check_qaer4["
                  << i << "][" << j << "]epsilon(" << epsilon << ").scale("
                  << scale << "): " << std::setprecision(14) << qaer4[i][j]
                  << " != " << check_qaer4[i][j] << std::endl;
      REQUIRE(qaer4[i][j] ==
              Approx(check_qaer4[i][j]).epsilon(epsilon).scale(scale));
    }
  }
  for (int i = 0; i < num_aerosol_ids; ++i) {
    for (int j = 0; j < num_modes; ++j) {
      for (int k = 0; k < nqtendaa; ++k) {
#ifdef HAERO_DOUBLE_PRECISION
        const double epsilon = 0.001;
#else
        const double epsilon = 1.0;
#endif
        const double scale = std::abs(qaer_delaa[i][j][k]);
        if (!(qaer_delaa[i][j][k] ==
              Approx(check_qaer_delaa[i][j][k]).epsilon(epsilon).scale(scale)))
          std::cout << "qaer_delaa[i][j][k] != "
                       "Approx(check_qaer_delaa[i][j][k]).epsilon("
                    << epsilon << ").scale(" << scale
                    << "): " << std::setprecision(14) << qaer_delaa[i][j][k]
                    << " != " << check_qaer_delaa[i][j][k] << std::endl;
        REQUIRE(
            qaer_delaa[i][j][k] ==
            Approx(check_qaer_delaa[i][j][k]).epsilon(epsilon).scale(scale));
      }
    }
  }
  for (int i = 0; i < num_modes; ++i) {
    if (!(qwtr4[i] == Approx(check_qwtr4[i])))
      std::cout << "qwtr4[i] != Approx(check_qwtr4[i])): "
                << std::setprecision(14) << qwtr4[i] << " != " << check_qwtr4[i]
                << std::endl;
    REQUIRE(qwtr4[i] == Approx(check_qwtr4[i]));
  }
}

TEST_CASE("mam_amicphys_1gridcell", "test_mam4_amicphys") {
  bool do_cond = {};
  bool do_rename = {};
  bool do_newnuc = {};
  bool do_coag = {};
  int nstep = {};
  Real deltat = {};
  int nsubarea = {};
  int ncldy_subarea = {};
  bool iscldy_subarea[maxsubarea] = {};
  Real afracsub[maxsubarea] = {};
  Real temp = {};
  Real pmid = {};
  Real pdel = {};
  Real zmid = {};
  Real pblh = {};
  Real relhumsub[maxsubarea] = {};
  Real dgn_a[AeroConfig::num_modes()] = {};
  Real dgn_awet[AeroConfig::num_modes()] = {};
  Real wetdens[AeroConfig::num_modes()] = {};
  Real qsub1[AeroConfig::num_gas_ids()][maxsubarea] = {};
  Real qsub2[AeroConfig::num_gas_ids()][maxsubarea] = {};
  Real qqcwsub2[AeroConfig::num_gas_ids()][maxsubarea] = {};
  Real qsub3[AeroConfig::num_gas_ids()][maxsubarea] = {};
  Real qqcwsub3[AeroConfig::num_gas_ids()][maxsubarea] = {};
  Real qaerwatsub3[AeroConfig::num_modes()][maxsubarea] = {};
  Real qsub4[AeroConfig::num_gas_ids()][maxsubarea] = {};
  Real qqcwsub4[AeroConfig::num_gas_ids()][maxsubarea] = {};
  Real qaerwatsub4[AeroConfig::num_modes()][maxsubarea] = {};
  Real qsub_tendaa[AeroConfig::num_gas_ids()][nqtendaa][maxsubarea] = {};
  Real qqcwsub_tendaa[AeroConfig::num_gas_ids()][nqqcwtendaa][maxsubarea] = {};

  // --------------------------------------------------------------------------------
  mam_amicphys_1gridcell(do_cond, do_rename, do_newnuc, do_coag, nstep, deltat,
                         nsubarea, ncldy_subarea, iscldy_subarea, afracsub,
                         temp, pmid, pdel, zmid, pblh, relhumsub, dgn_a,
                         dgn_awet, wetdens, qsub1, qsub2, qqcwsub2, qsub3,
                         qqcwsub3, qaerwatsub3, qsub4, qqcwsub4, qaerwatsub4,
                         qsub_tendaa, qqcwsub_tendaa);
}

TEST_CASE("modal_aero_amicphys_intr", "test_mam4_amicphys") {

  const int mdo_gasaerexch = 1;
  const int mdo_rename = 1;
  const int mdo_newnuc = 1;
  const int mdo_coag = 1;
  const int ncol = 1;
  const int nstep = 1;
  const Real deltat = 1;
  const Real t = 273.0;
  const Real pmid = 100000.0;
  const Real pdel = 0;
  const Real zm = 500.0;
  const Real pblh = 1100.0;
  const Real qv = 2.6351112225030129e-003;
  const Real cld = 0.4;
  // clang-format off
  Real q[gas_pcnst] =
    {0.0000000000000000, 1.9197285428799817e-011, 4.5213596233813260e-005, 0.0000000000000000, 1.2058113396053620e-009,
     3.4543891238118811e-011, 1.1035038530428116e-010, 3.3105115591284347e-010, 0.0000000000000000, 0.0000000000000000,
     6.8037089632647997e-011, 0.0000000000000000, 2253176148.8415728, 2.6039003387624210e-012, 2.4954461878281674e-011,
     6.8381333387002338e-012, 0.0000000000000000, 22531761488.415726, 0.0000000000000000, 5.4525090614694077e-010,
     1.3841779609564252e-010, 0.0000000000000000, 6.6326302211438671e-010, 1.9897890663431605e-009, -2.6523783784846809e-029,
     2253176.1488415725, 2.3663387458543536e-011, 9.4653549834174144e-011, 0.0000000000000000, 4506352297.6831455};
  Real qqcw[gas_pcnst] = {};
  const Real q_pregaschem[gas_pcnst] =
        {0.0000000000000000, 1.9196785428799816e-011, 4.5213596233813260e-005, 0.0000000000000000, 1.2058113396053620e-009,
         3.4543891238118811e-011, 1.1035038530428116e-010, 3.3105115591284347e-010, 0.0000000000000000, 0.0000000000000000,
         6.8037089632647997e-011, 0.0000000000000000, 2253176148.8415728, 2.6039003387624210e-012, 2.4954461878281674e-011,
         6.8381333387002338e-012, 0.0000000000000000, 22531761488.415726, 0.0000000000000000, 5.4525090614694077e-010,
         1.3841779609564252e-010, 0.0000000000000000, 6.6326302211438671e-010, 1.9897890663431605e-009, -2.6523783784846809e-029,
         2253176.1488415725, 2.3663387458543536e-011, 9.4653549834174144e-011, 0.0000000000000000, 4506352297.6831455};
  const Real q_precldchem[gas_pcnst] =
       { 0.0000000000000000, 1.9197285428799817e-011, 4.5213596233813260e-005, 0.0000000000000000, 1.2058113396053620e-009,
         3.4543891238118811e-011, 1.1035038530428116e-010, 3.3105115591284347e-010, 0.0000000000000000, 0.0000000000000000,
         6.8037089632647997e-011, 0.0000000000000000, 2253176148.8415728, 2.6039003387624210e-012, 2.4954461878281674e-011,
         6.8381333387002338e-012, 0.0000000000000000, 22531761488.415726, 0.0000000000000000, 5.4525090614694077e-010,
         1.3841779609564252e-010, 0.0000000000000000, 6.6326302211438671e-010, 1.9897890663431605e-009, -2.6523783784846809e-029,
         2253176.1488415725, 2.3663387458543536e-011, 9.4653549834174144e-011, 0.0000000000000000, 4506352297.6831455};
  const Real qqcw_precldchem[gas_pcnst] = {};
  Real q_tendbb[gas_pcnst][nqtendbb] = {};
  Real qqcw_tendbb[gas_pcnst][nqtendbb] = {};
  Real dgncur_a[AeroConfig::num_modes()] =
     {1.1944240222260541e-007, 2.5790443175784844e-008, 2.1718927087931639e-006, 5.2939491368692156e-008};
  Real dgncur_awet[AeroConfig::num_modes()] =
     {1.2705960055085835e-007, 2.7734941184652717e-008, 2.3266884562632178e-006, 5.2963714752164382e-008};
  Real wetdens_host[AeroConfig::num_modes()] =
     {1311.2951371145170, 1380.1581535473508, 1310.8217483170163, 1490.5543769911476};
  Real qaerwat[AeroConfig::num_modes()] =
     {6.7869256482212516e-011, 5.7791168928194034e-012, 4.5651595092659412e-010, 4.5048502047402045e-014};
  // clang-format on
  modal_aero_amicphys_intr(mdo_gasaerexch, mdo_rename, mdo_newnuc, mdo_coag,
                           ncol, nstep, deltat, t, pmid, pdel, zm, pblh, qv,
                           cld, q, qqcw, q_pregaschem, q_precldchem,
                           qqcw_precldchem, q_tendbb, qqcw_tendbb, dgncur_a,
                           dgncur_awet, wetdens_host, qaerwat);
}
