// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include "Kokkos_Core.hpp"
#include <mam4xx/aero_modes.hpp>
#include <mam4xx/gasaerexch.hpp>
#include <mam4xx/mam4.hpp>
#include <mam4xx/mode_wet_particle_size.hpp>
#include <validation.hpp>

#include <haero/aero_process.hpp>
#include <haero/atmosphere.hpp>
#include <haero/constants.hpp>

#include <cmath>
#include <ekat/ekat_assert.hpp>
#include <iostream>
#include <skywalker.hpp>

using namespace skywalker;
using namespace haero;
using namespace mam4;
// -------------------------------------------------------------------------

// This file contains a program for testing the gas-aerosol mass exchange
// parameterizations in the MAM4 aerosol microphysics packages.
//
// MAM4's function being called here is mam_gasaerexch_1subarea, which calls
// child functions to do time stepping for
//  1. the condensation of non-volatile species, e.g., H2SO4,
//  2. the condensation and evaporation of semi-volatile species, e.g., SOA.
//
// Depending on how the case is setup (what values are given to input
// parameters), this same test driver can be used to test the two types of
// gas-aerosol exchanges listed above, either in isolation or combined.
//
// The case setup (parameters, initial conditions, ambient conditions)
// is read in from a yaml file and parsed by a tool called Skywalker.
// Output is written into a Python module.
//
// Program skywkr_gasaerexch_timestepping below is the test driver.
// The namespace driver_utils contains some utility functions that are not
// specific to this test.
// -------------------------------------------------------------------------
// History:
// - Test ideas originally developed by Richard (Dick) C. Easter, PNNL, ca. 2017
// - Implementation using Skywalker and revisions to test setup: Qiyang Yan,
// PNNL, 2021-2022
// - Clean-up, consolidation, and further revision of test setup: Hui Wan, PNNL,
// 2022
// ------------------------------------------------------------------------

void usage() {
  std::cerr << "exe_skywkr_gasaerexch_timestepping: a Skywalker driver for "
               "validating the "
               "MAM4 nucleation parameterizations."
            << std::endl;
  std::cerr << "exe_skywkr_gasaerexch_timestepping: usage:" << std::endl;
  std::cerr << "exe_skywkr_gasaerexch_timestepping <input.yaml>" << std::endl;
  exit(0);
}

// -------------------------------------
void get_file_names(const std::string &input_suffix,
                    const std::string &output_suffix,
                    const std::string &input_file, std::string &output_file,
                    std::string &exe_name) {

  // Retrieve name of executable
  exe_name = "skywkr_gasaerexch_timestepping";

  // Generate an output file name based on the name of the input file.
  const size_t suffix = input_file.find(input_suffix);
  if (suffix != std::string::npos) {
    // Look for the last slash in the filename.
    size_t slash = input_file.rfind("/");
    if (slash == std::string::npos)
      slash = 0;
    output_file =
        input_file.substr(slash + 1, suffix - slash - 1) + output_suffix;
  } else {
    std::cout << exe_name << ": Invalid input filename (no " << input_suffix
              << " suffix found): " << input_file << std::endl;
    exit(1);
  }
  std::cout << __FILE__ << ":" << __LINE__ << std::endl
            << " input_suffix:" << input_suffix << std::endl
            << " output_suffix:" << output_suffix << std::endl
            << " input_file:" << input_file << std::endl
            << " output_file:" << output_file << std::endl;
}

// ------------------------------------------------------------
// Subroutine for initialize various module constants/parameters.
// In the global model, the initialization of these variables is
// scattered in multiple subroutines and some values are copied
// from the host model. Here we have a simplified subroutine that
// only set the constants/parameters needed by the gas-aerosol exchange
// parameterization.
// ------------------------------------------------------------
GasAerExch::Config gasaerexch_module_var_init() {
  const int imode_pca = static_cast<int>(ModeIndex::PrimaryCarbon);
  static constexpr int max_mode = 4;
  int mode_aging_optaa[max_mode] = {};
  mode_aging_optaa[imode_pca] =
      1; // the value does not matter as long as it is > 0
  bool l_mode_can_age[max_mode] = {};
  for (int i = 0; i < max_mode; ++i)
    l_mode_can_age[i] = mode_aging_optaa[i] > 0;

  GasAerExch::Config config;
  for (int i = 0; i < GasAerExch::num_mode; ++i)
    config.l_mode_can_age[i] = l_mode_can_age[i];

  return config;
}

// =================================================================
//  This is the driver program that tests MAM's gas-aerosol exchange
//  parameterizations.
// =================================================================

int main(int argc, char **argv) {
  if (argc == 1) {
    usage();
  }
  validation::initialize(argc, argv);
  std::string input_file = argv[1];
  std::cout << argv[0] << ": reading " << input_file << std::endl;

  static constexpr int num_gas = mam4::AeroConfig::num_gas_ids();
  static constexpr int num_mode = mam4::AeroConfig::num_modes();
  static constexpr int num_aer = mam4::AeroConfig::num_aerosol_ids();
  static constexpr int ntot_amode = 4;

  // ---------------------------------------------------------------------
  // Initialize constants, parameters, and MAMs internal bookkeeping
  // --------------------------------------------------------------------
  const GasAerExch::Config config = gasaerexch_module_var_init();
  // molecular_weight is not the same as from mam4::aero_species()
  // even when converted from kg to gm.  So hard code them in this
  // test to be able to match that which is in mam_refactor
  const Real molecular_weight_gm[num_aer] = {150.0, 115.0, 150.0,   12.0,
                                             58.50, 135.0, 250092.0};

  // --------------------------------------------------------------------
  // Read command line, retrieve names of executable and input file, set name of
  // output file
  // --------------------------------------------------------------------

  const std::string input_suffix = ".yaml";
  const std::string output_suffix = ".py";
  std::string output_file;
  std::string exe_name;
  get_file_names(input_suffix, output_suffix, input_file, output_file,
                 exe_name);
  // -------------------------------------------------------
  // Load the ensemble. Any error encountered is fatal.
  // -------------------------------------------------------
  std::cout << exe_name << ": Loading ensemble from " << input_file
            << std::endl;
  const std::string model_name = "mam_box";
  Ensemble *ensemble = skywalker::load_ensemble(input_file, model_name);

  // ----------------------------------------------------------------------
  //  Parse test settings. Make sure the subroutine to be tested is
  //  mam_gasaerexch_1subarea
  // ----------------------------------------------------------------------
  Settings settings = ensemble->settings();
  if (!settings.has("mam_subr_name")) {
    std::cerr << "No function specified in mam4xx.mam_subr_name!" << std::endl;
    exit(1);
  }
  if (settings.get("mam_subr_name") != "mam_gasaerexch_1subarea") {
    std::cerr << "mam4xx.mam_subr_name not `mam_gasaerexch_1subarea` "
              << std::endl;
    exit(1);
  }

  const int h2so4 = static_cast<int>(mam4::GasId::H2SO4);

  const int soa = static_cast<int>(mam4::AeroId::SOA);
  const int so4 = static_cast<int>(mam4::AeroId::SO4);
  const int pom = static_cast<int>(mam4::AeroId::POM);
  const int bc = static_cast<int>(mam4::AeroId::BC);
  const int ncl = static_cast<int>(mam4::AeroId::NaCl);
  const int dst = static_cast<int>(mam4::AeroId::DST);
  const int mom = static_cast<int>(mam4::AeroId::MOM);
  // =======================================================================
  //  Loop over all members of the ensemble. Process input, do calculations, and
  //  prepare output
  // =======================================================================
  ensemble->process([=](const Input &input, Output &output) {
    Real qgas_netprod_otrproc[num_gas];
    Real qgas_cur[num_mode];
    // ----------------------------------------
    //  Process input for this ensemble member
    // -----------------------------------saerexch_module_va-----
    //  Ambient conditions

    const Real pmid = input.get("pmid"); // air pressure
    const Real temp = input.get("temp"); // air temperature
    // const Real aircon = pmid/(Constants::r_gas*temp); // air density

    // gas production rates and mixing ratio ICs

    qgas_netprod_otrproc[h2so4] = input.get("qgas_prod_rate_h2so4");
    qgas_netprod_otrproc[soa] = input.get("qgas_prod_rate_soag");

    qgas_cur[h2so4] = input.get("qgas_cur_h2so4");
    qgas_cur[soa] = input.get("qgas_cur_soag");

    // aerosol mixing ratio ICs
    Real qnum_cur[num_mode];
    {
      const std::vector<Real> val = input.get_array("qnum_cur");
      for (int i = 0; i < num_mode; ++i)
        qnum_cur[i] = val[i];
    }
    Real qaer_cur[num_aer][num_mode];
    {
      std::vector<Real> val[7];
      val[0] = input.get_array("qaer_soa");
      val[1] = input.get_array("qaer_so4");
      val[2] = input.get_array("qaer_pom");
      val[3] = input.get_array("qaer_bc");
      val[4] = input.get_array("qaer_ncl");
      val[5] = input.get_array("qaer_dst");
      val[6] = input.get_array("qaer_mom");
      for (int i = 0; i < num_mode; ++i) {
        qaer_cur[soa][i] = val[0][i];
        qaer_cur[so4][i] = val[1][i];
        qaer_cur[pom][i] = val[2][i];
        qaer_cur[bc][i] = val[3][i];
        qaer_cur[ncl][i] = val[4][i];
        qaer_cur[dst][i] = val[5][i];
        qaer_cur[mom][i] = val[6][i];
      }
    }
    // Time-stepping
    const Real run_length = input.get("run_length");
    const Real dt_mam = input.get("dt_mam");
    int nstep_end = std::round(run_length / dt_mam);
    EKAT_REQUIRE_MSG(
        mam4::FloatingPoint<Real>::equiv(nstep_end * dt_mam, run_length),
        "The run length should be a multiple of the time step.");
    ++nstep_end;

    const Real dt_soa_opt = std::round(input.get("dt_soa_opt"));
    EKAT_REQUIRE_MSG(dt_soa_opt == 0 || dt_soa_opt == -1,
                     "dt_soa_opt should be 0 or -1.");
    const Real dt_soa_fixed = dt_soa_opt == -1 ? -1 : dt_mam;

    const bool update_diameter_every_time_step =
        std::round(input.get("update_diameter_every_time_step"));

    // Miscellaneous input and tmp variables

    const int n_mode = ntot_amode - 1;
    const Real dwet_ddry_ratio = 1.0;
    // const bool l_calc_gas_uptake_coeff = true;

    std::vector<Real> time(nstep_end, 0);
    std::vector<Real> so4a(nstep_end, 0);
    std::vector<Real> so4g(nstep_end, 0);
    std::vector<Real> so4g_ddt_exch(nstep_end, 0);
    std::vector<Real> soaa(nstep_end, 0);
    std::vector<Real> soag(nstep_end, 0);
    std::vector<Real> soag_ddt_exch(nstep_end, 0);
    std::vector<Real> soag_amb_qsat(nstep_end, 0);
    std::vector<Real> soag_niter(nstep_end, 0);

    // ------------------------------------------------------------------
    //  save initial conditions for output
    // ------------------------------------------------------------------
    {
      const int istep = 0;
      time[istep] = 0;
      so4g[istep] = qgas_cur[h2so4];
      soag[istep] = qgas_cur[soa];
      for (int i = 0; i < n_mode; ++i)
        so4a[istep] += qaer_cur[so4][i];
      for (int i = 0; i < n_mode; ++i)
        soaa[istep] += qaer_cur[soa][i];
      // ------------------------------------------------------------------
      //  Time loop, in which the MAM subroutine we want to test is called.
      // ------------------------------------------------------------------
      std::cout << std::endl;
      std::cout << "===== Time loop starting" << std::endl;
      std::cout << "run_length   = " << run_length << std::endl;
      std::cout << "nstep_end    = " << nstep_end << std::endl;
      std::cout << "dt_mam       = " << dt_mam << std::endl;
      std::cout << "dt_soa_fixed = " << dt_soa_fixed
                << " (dt_soa_opt = " << dt_soa_opt << ")" << std::endl;
      std::cout << std::endl;
    }

    // create containers for the timestepping below.
    int nlev = 1;
    Real pblh = 1000;
    Atmosphere atm = validation::create_atmosphere(nlev, pblh);
    mam4::Prognostics progs = validation::create_prognostics(nlev);
    mam4::Diagnostics diags = validation::create_diagnostics(nlev);
    mam4::Tendencies tends = validation::create_tendencies(nlev);

    // ---------
    for (int istep = 1; istep < nstep_end; ++istep) {

      // ------------------------------------------------------------------
      //  Apply net production from other processes (e.g., chemistry,
      //  advection). This is done here for SOA only, because the production
      //  rate of H2SO4 gas is handled inside the MAM subroutine we are testing.
      // ------------------------------------------------------------------
      qgas_cur[soa] = qgas_cur[soa] + qgas_netprod_otrproc[soa] * dt_mam;

      // ------------------------------------------------------------------
      //  Calculate/update wet geometric mean diameter of each aerosol mode
      // ------------------------------------------------------------------
      // geometric mean diameter of each aerosol mode
      Real dgn_awet[num_mode] = {};
      if ((update_diameter_every_time_step == 1) || (istep == 1)) {
        mam4::diag_dgn_wet(qaer_cur, qnum_cur, molecular_weight_gm,
                           dwet_ddry_ratio, dgn_awet);
      }
      // ------------------------------------------------------------------
      //  Gas-aerosol exchanges
      // ------------------------------------------------------------------
      //  Save the gas mixing ratios before gas-aerosol exchange to diagnose the
      //  tendencies after the subroutine call.
      // Note: qgas_cur[num_gas] is out of bounds but the constructor is for
      // [begin,end)
      const std::vector<Real> zqgas_bef(&qgas_cur[0], &qgas_cur[num_mode]);

      // Call the MAM subroutine. Gas and aerosol mixing ratios will be updated
      // during the call
      Kokkos::deep_copy(atm.temperature, temp);
      Kokkos::deep_copy(atm.pressure, pmid);
      for (int n = 0; n < num_mode; ++n)
        for (int g = 0; g < num_aer; ++g)
          Kokkos::deep_copy(progs.q_aero_i[n][g], qaer_cur[g][n]);

      for (int n = 0; n < num_gas; ++n)
        Kokkos::deep_copy(progs.q_gas[n], qgas_cur[n]);

      for (int n = 0; n < num_mode; ++n)
        Kokkos::deep_copy(progs.n_mode_i[n], qnum_cur[n]);

      for (int igas = 0; igas < num_gas; ++igas)
        for (int imode = 0; imode < num_mode; ++imode)
          Kokkos::deep_copy(progs.uptkaer[igas][imode], 0);

      Kokkos::deep_copy(diags.g0_soa_out, 0);

      for (int i = 0; i < num_mode; ++i)
        Kokkos::deep_copy(diags.wet_geometric_mean_diameter_i[i], dgn_awet[i]);

      mam4::AeroConfig mam4_config;

      mam4::GasAerExchProcess::ProcessConfig process_config = config;
      for (int i = 0; i < num_gas; ++i)
        process_config.qgas_netprod_otrproc[i] = qgas_netprod_otrproc[i];

      mam4::GasAerExchProcess process(mam4_config, process_config);
      auto team_policy = ThreadTeamPolicy(1u, Kokkos::AUTO);
      Real t = 0.0, dt = dt_mam;
      Kokkos::parallel_for(
          team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
            process.compute_tendencies(team, t, dt, atm, progs, diags, tends);
          });

      // ---------------------------------------------------------
      //  Calculations for this timestep done. Prepare for output.
      // ---------------------------------------------------------
      //  Save values for output and postprocessing

      for (int n = 0; n < num_mode; ++n)
        for (int g = 0; g < num_aer; ++g) {
          auto host_view = Kokkos::create_mirror_view(progs.q_aero_i[n][g]);
          Kokkos::deep_copy(host_view, progs.q_aero_i[n][g]);
          qaer_cur[g][n] = host_view(0);
        }

      for (int n = 0; n < num_gas; ++n) {
        auto host_view = Kokkos::create_mirror_view(progs.q_gas[n]);
        Kokkos::deep_copy(host_view, progs.q_gas[n]);
        qgas_cur[n] = host_view(0);
      }

      // ambient saturation mixing ratio of SOA gases, solute effect ignored
      Real g0_soa = 0;
      {
        auto host_view = Kokkos::create_mirror_view(diags.g0_soa_out);
        Kokkos::deep_copy(host_view, diags.g0_soa_out);
        g0_soa = host_view(0);
      }

      // Number of time substeps needed for convergence of SOA gases
      int niter = 0;
      {
        auto host_view = Kokkos::create_mirror_view(diags.num_substeps);
        Kokkos::deep_copy(host_view, diags.num_substeps);
        niter = host_view(0);
      }

      time[istep] = istep * dt_mam;

      // so4
      so4g[istep] = qgas_cur[h2so4];
      so4a[istep] = 0;
      for (int n = 0; n < num_mode; ++n)
        so4a[istep] += qaer_cur[so4][n];

      so4g_ddt_exch[istep] = (qgas_cur[h2so4] - zqgas_bef[h2so4]) / dt_mam -
                             qgas_netprod_otrproc[h2so4];

      // soa
      soag[istep] = qgas_cur[soa];
      soaa[istep] = 0;
      for (int n = 0; n < num_mode; ++n)
        soaa[istep] += qaer_cur[soa][n];
      soag_ddt_exch[istep] = (qgas_cur[soa] - zqgas_bef[soa]) / dt_mam;
      soag_amb_qsat[istep] = g0_soa;
      soag_niter[istep] = niter;
      // Print some numbers to stdout for quick checks
      std::cout << "step " << istep << ", dqSOAG/dt = " << soag_ddt_exch[istep]
                << ", qSOAG after = " << qgas_cur[soa]
                << ", g0_soa = " << g0_soa << ", niter = " << niter
                << std::endl;
      // ----------------------------------------------
    }
    std::cout << std::endl;
    std::cout << "===== Time loop done" << std::endl;
    std::cout << std::endl;

    // ------------------------------------------
    //  Process output for this ensemble member
    // ------------------------------------------
    output.set("model_time", time);
    output.set("so4a_time_series", so4a);
    output.set("so4g_time_series", so4g);
    output.set("so4g_ddt_exch", so4g_ddt_exch);
    output.set("soaa_time_series", soaa);
    output.set("soag_time_series", soag);
    output.set("soag_ddt_exch", soag_ddt_exch);
    output.set("soag_amb_qsat", soag_amb_qsat);
    output.set("soag_niter", soag_niter);
  });

  // ========================================
  //  End of loop over all ensemble members
  // ========================================//
  //  Now we write out a Python module containing the output data.
  std::cout << exe_name << ": Writing output to " << output_file << std::endl;
  ensemble->write(output_file);

  // If we got here, the execution was successfull.
  std::cout << "PASS" << std::endl;
  validation::finalize();
  return 0;
}
