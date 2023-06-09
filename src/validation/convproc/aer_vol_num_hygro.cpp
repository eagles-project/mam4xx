// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <ekat/ekat_assert.hpp>
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
  EKAT_ASSERT(host.size() == size);
  dev = mam4::validation::create_column_view(size);
  auto host_view = Kokkos::create_mirror_view(dev);
  for (int n = 0; n < size; ++n)
    host_view[n] = host[n];
  Kokkos::deep_copy(dev, host_view);
}
void set_output(Output &output, const std::string &name, const int size,
                std::vector<Real> &host, const ColumnView &dev) {
  host.resize(size);
  auto host_view = Kokkos::create_mirror_view(dev);
  Kokkos::deep_copy(host_view, dev);
  for (int n = 0; n < size; ++n)
    host[n] = host_view[n];
  output.set(name, host);
}
} // namespace
void aer_vol_num_hygro(Ensemble *ensemble) {
  // We don't need any settings for this particular test.
  // Settings settings = ensemble->settings();
  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    const int num_modes = AeroConfig::num_modes();
    // Fetch ensemble parameters
    // Convert to C++ index by subtracting one.
    const int pcnst_extd = input.get("pcnst_extd");
    EKAT_ASSERT(pcnst_extd == ConvProc::pcnst_extd);
    const Real rhoair = input.get("rhoair");

    std::vector<Real> conu_host, vaerosol_host, naerosol_host, hygro_host,
        hygro_2_host;
    ColumnView conu_dev, vaerosol_dev, naerosol_dev, hygro_dev, hygro_2_dev;
    get_input(input, "conu", ConvProc::pcnst_extd, conu_host, conu_dev);
    vaerosol_dev = mam4::validation::create_column_view(num_modes);
    naerosol_dev = mam4::validation::create_column_view(num_modes);
    hygro_dev = mam4::validation::create_column_view(num_modes);
    hygro_2_dev = mam4::validation::create_column_view(num_modes);
    Kokkos::parallel_for(
        "aer_vol_num_hygro", 1, KOKKOS_LAMBDA(int) {
          Real conu[ConvProc::pcnst_extd];
          for (int i = 0; i < ConvProc::pcnst_extd; ++i)
            conu[i] = conu_dev[i];
          Real vaerosol[num_modes];
          Real naerosol[num_modes];
          Real hygro[num_modes];
          convproc::aer_vol_num_hygro(conu, rhoair, vaerosol, naerosol, hygro);
          for (int i = 0; i < num_modes; ++i)
            vaerosol_dev[i] = vaerosol[i];
          for (int i = 0; i < num_modes; ++i)
            naerosol_dev[i] = naerosol[i];
          for (int i = 0; i < num_modes; ++i)
            hygro_dev[i] = hygro[i];
          for (int i = 0; i < ConvProc::pcnst_extd; ++i)
            conu[i] /= 1.0e21;
          convproc::aer_vol_num_hygro(conu, rhoair, vaerosol, naerosol, hygro);
          for (int i = 0; i < num_modes; ++i)
            hygro_2_dev[i] = hygro[i];
        });
    // Check case of iflux_method == 2 which is not part of the e3sm tests.
    set_output(output, "vaerosol", num_modes, vaerosol_host, vaerosol_dev);
    set_output(output, "naerosol", num_modes, naerosol_host, naerosol_dev);
    set_output(output, "hygro", num_modes, hygro_host, hygro_dev);
    hygro_2_host.resize(num_modes);
    {
      auto host_view = Kokkos::create_mirror_view(hygro_2_dev);
      Kokkos::deep_copy(host_view, hygro_2_dev);
      for (int n = 0; n < num_modes; ++n)
        hygro_2_host[n] = host_view[n];
    }
    // This special test checks an if statement in aer_vol_num_hygro that
    // sets hygro to 0.2 in case of very small volume. It is tripped twice:
    EKAT_ASSERT(std::abs(hygro_host[0] - hygro_2_host[0]) < .0000001);
    EKAT_ASSERT(std::abs(hygro_host[1] - hygro_2_host[1]) < .0000001);
    EKAT_ASSERT(0.2 == hygro_2_host[2]);
    EKAT_ASSERT(0.2 == hygro_2_host[3]);
  });
}
