// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include "Kokkos_Core.hpp"
#include <catch2/catch.hpp>
#include <iomanip>
#include <iostream>
#include <mam4xx/wet_dep.hpp>
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
void test_compute_tendencies(std::unique_ptr<Ensemble> &ensemble) {
  // We don't need any settings for this particular test.
  // Settings settings = ensemble->settings();
  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    const int nlev = 72;
    const Real t = 0;
    const Real dt = 36000;
    const Real pblh = 1000;
    const int gas_pcnst = 40;
    EKAT_ASSERT(input.get("dt") == 3600);

    Atmosphere atmosphere = validation::create_atmosphere(nlev, pblh);
    Surface surface = validation::create_surface();
    mam4::Prognostics prognostics = validation::create_prognostics(nlev);
    mam4::Diagnostics diagnostics = validation::create_diagnostics(nlev);
    mam4::Tendencies tendencies = validation::create_tendencies(nlev);
    mam4::AeroConfig aero_config;
    mam4::WetDeposition::Config wetdep_config;

    mam4::WetDeposition wetdep;
    wetdep.init(aero_config, wetdep_config);

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
    diagnostics.evaporation_of_falling_precipitation =
        mam4::validation::create_column_view(nlev);
    diagnostics.aerosol_wet_deposition_interstitial =
        mam4::validation::create_column_view(nlev);
    diagnostics.aerosol_wet_deposition_cloud_water =
        mam4::validation::create_column_view(nlev);
    for (int i = 0; i < AeroConfig::num_modes(); ++i)
      diagnostics.wet_geometric_mean_diameter_i[i] =
          mam4::validation::create_column_view(nlev);
    auto mixing_ratio = mam4::validation::create_column_view(nlev * gas_pcnst);
    diagnostics.tracer_mixing_ratio =
        Diagnostics::ColumnTracerView(mixing_ratio.data(), nlev, gas_pcnst);
    auto mixing_ratio_dt =
        mam4::validation::create_column_view(nlev * gas_pcnst);
    diagnostics.d_tracer_mixing_ratio_dt =
        Diagnostics::ColumnTracerView(mixing_ratio_dt.data(), nlev, gas_pcnst);
    Kokkos::parallel_for(
        "init_column_views", nlev, KOKKOS_LAMBDA(int i) {
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
          diagnostics.evaporation_of_falling_precipitation[i] = 0;
          diagnostics.aerosol_wet_deposition_interstitial[i] = 0;
          diagnostics.aerosol_wet_deposition_cloud_water[i] = 0;
          for (int j = 0; j < gas_pcnst; ++j) {
            diagnostics.tracer_mixing_ratio(i, j) = 0;
            diagnostics.d_tracer_mixing_ratio_dt(i, j) = 0;
          }
        });
    Kokkos::fence();
    std::vector<Real> temperature_host, pdel_host, pmid_host, cldfrac_host,
        icwmr_host, rprd_host, evapc_host, dqdt_host, dp_host, dgn_awet_host,
        evapr_host;
    ColumnView temperature_dev, pmid_dev, pdel_dev;

    get_input(input, "state_t", nlev, temperature_host, temperature_dev);
    atmosphere.temperature = temperature_dev;
    get_input(input, "state_pmid", nlev, pmid_host, pmid_dev);
    atmosphere.pressure = pmid_dev;
    get_input(input, "state_pdel", nlev, pdel_host, pdel_dev);
    atmosphere.hydrostatic_dp = pdel_dev;
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
    get_input(input, "qnew", nlev, gas_pcnst, dp_host,
              diagnostics.tracer_mixing_ratio);
    get_input(input, "evapr", nlev, evapr_host,
              diagnostics.evaporation_of_falling_precipitation);
    ColumnView awet_tmp;
    get_input(input, "dgn_awet", AeroConfig::num_modes(), dgn_awet_host,
              awet_tmp);
    for (int i = 0; i < AeroConfig::num_modes(); ++i) {
      auto host_view = Kokkos::create_mirror_view(
          diagnostics.wet_geometric_mean_diameter_i[i]);
      for (int j = 0; j < nlev; ++j)
        host_view[j] = dgn_awet_host[i];
      Kokkos::deep_copy(diagnostics.wet_geometric_mean_diameter_i[i],
                        host_view);
    }
    // NOTE: we haven't parallelized wetdep over vertical levels because of
    // NOTE: data dependencies, so we run this serially
    auto team_policy = haero::ThreadTeamPolicy(1u, 1u);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
          wetdep.compute_tendencies(aero_config, team, t, dt, atmosphere,
                                    surface, prognostics, diagnostics,
                                    tendencies);
        });
    Kokkos::fence();
    set_output(output, "dqdt", nlev, gas_pcnst, dqdt_host,
               diagnostics.d_tracer_mixing_ratio_dt);
  });
}
