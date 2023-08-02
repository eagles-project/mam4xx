// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include "Kokkos_Core.hpp"
#include <catch2/catch.hpp>
#include <iomanip>
#include <iostream>
#include <mam4xx/convproc.hpp>
#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;

namespace {
void get_input(const Input &input, const std::string &name, const int size,
               std::vector<Real> &host, ColumnView &dev) {
  host = input.get_array(name);
  dev = mam4::validation::create_column_view(size);

  EKAT_ASSERT(host.size() == size);
  auto host_view = Kokkos::create_mirror_view(dev);
  for (int n = 0; n < size; ++n)
    host_view[n] = host[n];
  Kokkos::deep_copy(dev, host_view);
}
void get_input(const Input &input, const std::string &name, const int rows,
               const int cols, std::vector<Real> &host,
               Diagnostics::ColumnTracerView &dev) {
  host = input.get_array(name);
  ColumnView col_view = mam4::validation::create_column_view(rows * cols);
  dev = Diagnostics::ColumnTracerView(col_view.data(), rows, cols);
  EKAT_ASSERT(host.size() == rows * cols);
  {
    std::vector<std::vector<Real>> matrix(rows, std::vector<Real>(cols));
    for (int j = 0, n = 0; j < cols; ++j)
      for (int i = 0; i < rows; ++i, ++n)
        matrix[i][j] = host[n];
    auto host_view = Kokkos::create_mirror_view(dev);
    for (int i = 0; i < rows; ++i)
      for (int j = 0; j < cols; ++j)
        host_view(i, j) = matrix[i][j];
    Kokkos::deep_copy(dev, host_view);
  }
}
void set_output(Output &output, const std::string &name, const int rows,
                const int cols, std::vector<Real> &host,
                const Diagnostics::ColumnTracerView &dev) {
  host.resize(rows * cols);
  auto host_view = Kokkos::create_mirror_view(dev);
  Kokkos::deep_copy(host_view, dev);
  for (int j = 0, n = 0; j < cols; ++j)
    for (int i = 0; i < rows; ++i, ++n) {
      host[n] = host_view(i, j);
    }
  output.set(name, host);
}
} // namespace
void compute_tendencies(Ensemble *ensemble) {
  // We don't need any settings for this particular test.
  // Settings settings = ensemble->settings();
  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    const int nlev = 72;
    const int ktop = 47;
    const int kbot = 71;
    const Real t = 0;
    const Real dt = 36000;
    const Real pblh = 1000;
    // const int pcnst_extd = ConvProc::pcnst_extd;
    const int pcnst = ConvProc::gas_pcnst;
    // Fetch ensemble parameters
    // Convert to C++ index by subtracting one.
    // ktop is jt(il1g) to jt(il2g) in Fortran
    // but jt is just a scalar and il1g=il2g=48;
    // kbot is mx(il1g) to jt(il2g) in Fortran
    // but mx is just a scalar iand l1g=il2g=71;
    EKAT_ASSERT(input.get("dt") == 3600);
    EKAT_ASSERT(input.get("jt") == 72);

    int mmtoo_prevap_resusp[pcnst];
    {
      const int resusp[pcnst] = {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                                 0,  0,  0,  0,  0,  31, 33, 34, 32, 29,
                                 30, 35, -2, 31, 34, 30, 35, -2, 29, 30,
                                 31, 32, 33, 34, 35, -2, 33, 32, 35, -2};
      for (int i = 0; i < pcnst; ++i)
        mmtoo_prevap_resusp[i] = resusp[i] - 1;
    }

    Atmosphere atmosphere = validation::create_atmosphere(nlev, pblh);
    Surface surface = validation::create_surface();
    mam4::Prognostics prognostics = validation::create_prognostics(nlev);
    mam4::Diagnostics diagnostics = validation::create_diagnostics(nlev);
    mam4::Tendencies tendencies = validation::create_tendencies(nlev);
    mam4::AeroConfig aero_config;
    mam4::ConvProc::Config convproc_config;
    convproc_config.convproc_do_aer = true;
    convproc_config.convproc_do_gas = false;
    convproc_config.nlev = nlev;
    convproc_config.ktop = ktop;
    convproc_config.kbot = kbot;

    std::vector<Real> species_class_host;
    ColumnView species_class_dev;
    get_input(input, "species_class", pcnst, species_class_host,
              species_class_dev);
    for (int i = 0; i < pcnst; ++i)
      convproc_config.species_class[i] = species_class_host[i];
    for (int i = 0; i < pcnst; ++i)
      convproc_config.mmtoo_prevap_resusp[i] = mmtoo_prevap_resusp[i];

    mam4::ConvProc convproc;
    convproc.init(aero_config, convproc_config);

    diagnostics.hydrostatic_dry_dp = mam4::validation::create_column_view(nlev);
    diagnostics.deep_convective_cloud_fraction =
        mam4::validation::create_column_view(nlev);
    diagnostics.shallow_convective_cloud_fraction =
        mam4::validation::create_column_view(nlev);
    diagnostics.deep_convective_cloud_condensate =
        mam4::validation::create_column_view(nlev);
    diagnostics.shallow_convective_cloud_condensate =
        mam4::validation::create_column_view(nlev);
    diagnostics.deep_convective_precipitation_production =
        mam4::validation::create_column_view(nlev);
    diagnostics.shallow_convective_precipitation_production =
        mam4::validation::create_column_view(nlev);
    diagnostics.deep_convective_precipitation_evaporation =
        mam4::validation::create_column_view(nlev);
    diagnostics.shallow_convective_precipitation_evaporation =
        mam4::validation::create_column_view(nlev);
    diagnostics.total_convective_detrainment =
        mam4::validation::create_column_view(nlev);
    diagnostics.shallow_convective_detrainment =
        mam4::validation::create_column_view(nlev);
    diagnostics.shallow_convective_ratio =
        mam4::validation::create_column_view(nlev);
    diagnostics.mass_entrain_rate_into_updraft =
        mam4::validation::create_column_view(nlev);
    diagnostics.mass_entrain_rate_into_downdraft =
        mam4::validation::create_column_view(nlev);
    diagnostics.mass_detrain_rate_from_updraft =
        mam4::validation::create_column_view(nlev);
    diagnostics.delta_pressure = mam4::validation::create_column_view(nlev);
    auto mixing_ratio =
        mam4::validation::create_column_view(nlev * ConvProc::gas_pcnst);
    diagnostics.tracer_mixing_ratio = Diagnostics::ColumnTracerView(
        mixing_ratio.data(), nlev, ConvProc::gas_pcnst);
    auto mixing_ratio_dt =
        mam4::validation::create_column_view(nlev * ConvProc::gas_pcnst);
    diagnostics.d_tracer_mixing_ratio_dt = Diagnostics::ColumnTracerView(
        mixing_ratio_dt.data(), nlev, ConvProc::gas_pcnst);
    Kokkos::parallel_for(
        "init_column_views", nlev, KOKKOS_LAMBDA(int i) {
          diagnostics.hydrostatic_dry_dp[i] = 0;
          diagnostics.deep_convective_cloud_fraction[i] = 0;
          diagnostics.shallow_convective_cloud_fraction[i] = 0;
          diagnostics.deep_convective_cloud_condensate[i] = 0;
          diagnostics.shallow_convective_cloud_condensate[i] = 0;
          diagnostics.deep_convective_precipitation_production[i] = 0;
          diagnostics.shallow_convective_precipitation_production[i] = 0;
          diagnostics.deep_convective_precipitation_evaporation[i] = 0;
          diagnostics.shallow_convective_precipitation_evaporation[i] = 0;
          diagnostics.total_convective_detrainment[i] = 0;
          diagnostics.shallow_convective_detrainment[i] = 0;
          diagnostics.shallow_convective_ratio[i] = 0;
          diagnostics.mass_entrain_rate_into_updraft[i] = 0;
          diagnostics.mass_entrain_rate_into_downdraft[i] = 0;
          diagnostics.mass_detrain_rate_from_updraft[i] = 0;
          diagnostics.delta_pressure[i] = 0;
          for (int j = 0; j < ConvProc::gas_pcnst; ++j) {
            diagnostics.tracer_mixing_ratio(i, j) = 0;
            diagnostics.d_tracer_mixing_ratio_dt(i, j) = 0;
          }
        });
    Kokkos::fence();
    std::vector<Real> temperature_host, pmid_host, du_host, eu_host, ed_host,
        dp_host, dpdry_host, cldfrac_host, icwmr_host, rprd_host, evapc_host,
        dqdt_host;
    ColumnView temperature_dev, pmid_dev;

    get_input(input, "state_pdeldry", nlev, dpdry_host,
              diagnostics.hydrostatic_dry_dp);
    get_input(input, "state_t", nlev, temperature_host, temperature_dev);
    atmosphere.temperature = temperature_dev;
    get_input(input, "state_pmid", nlev, pmid_host, pmid_dev);
    atmosphere.pressure = pmid_dev;
    get_input(input, "dp_frac", nlev, cldfrac_host,
              diagnostics.deep_convective_cloud_fraction);
    get_input(input, "cldfrac", nlev, cldfrac_host,
              diagnostics.total_convective_detrainment);
    get_input(input, "icwmrdp", nlev, icwmr_host,
              diagnostics.deep_convective_cloud_condensate);
    get_input(input, "rprddp", nlev, rprd_host,
              diagnostics.deep_convective_precipitation_production);
    get_input(input, "evapcdp", nlev, evapc_host,
              diagnostics.deep_convective_precipitation_evaporation);
    get_input(input, "du", nlev, du_host,
              diagnostics.mass_detrain_rate_from_updraft);
    get_input(input, "eu", nlev, eu_host,
              diagnostics.mass_entrain_rate_into_updraft);
    get_input(input, "ed", nlev, ed_host,
              diagnostics.mass_entrain_rate_into_downdraft);
    get_input(input, "dp", nlev, dp_host, diagnostics.delta_pressure);
    get_input(input, "qnew", nlev, pcnst, dp_host,
              diagnostics.tracer_mixing_ratio);

    // NOTE: we haven't parallelized convproc over vertical levels because of
    // NOTE: data dependencies, so we run this serially
    auto team_policy = haero::ThreadTeamPolicy(1u, 1u);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
          convproc.compute_tendencies(aero_config, team, t, dt, atmosphere,
                                      prognostics, diagnostics, tendencies);
        });
    Kokkos::fence();
    set_output(output, "dqdt", nlev, pcnst, dqdt_host,
               diagnostics.d_tracer_mixing_ratio_dt);
  });
}
