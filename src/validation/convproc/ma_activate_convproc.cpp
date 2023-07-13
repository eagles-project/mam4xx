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
void ma_activate_convproc(Ensemble *ensemble) {
  // We don't need any settings for this particular test.
  // Settings settings = ensemble->settings();
  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    const int pcnst_extd = ConvProc::pcnst_extd;
    // Fetch ensemble parameters
    // Convert to C++ index by subtracting one.
    const Real f_ent = input.get("f_ent");
    EKAT_ASSERT(std::abs(f_ent - 0.3036903622e-01) < .000001);
    const Real dt_u = input.get("dt_u");
    EKAT_ASSERT(std::abs(dt_u - 0.7956304995e+03) < .000001);
    const Real wup = input.get("wup");
    EKAT_ASSERT(std::abs(wup - 0.3518837909e+01) < .000001);
    const Real tair = input.get("tair");
    EKAT_ASSERT(std::abs(tair - 0.2854111562e+03) < .000001);
    const Real rhoair = input.get("rhoair");
    EKAT_ASSERT(std::abs(rhoair - 0.9353223498e+00) < .000001);
    const int kk = input.get("kk") - 1;
    const int kactfirst = input.get("kactfirst") - 1;
    EKAT_ASSERT(kactfirst == 67);

    std::vector<Real> conu_host, dconudt_host;
    ColumnView conu_dev, dconudt_dev;
    get_input(input, "conu", pcnst_extd, conu_host, conu_dev);
    get_input(input, "dconudt", pcnst_extd, dconudt_host, dconudt_dev);
    Kokkos::parallel_for(
        "ma_activate_convproc", 1, KOKKOS_LAMBDA(int) {
          Real conu[pcnst_extd], dconudt[pcnst_extd];
          for (int i = 0; i < pcnst_extd; ++i)
            conu[i] = conu_dev[i];
          for (int i = 0; i < pcnst_extd; ++i)
            dconudt[i] = dconudt_dev[i];
          convproc::ma_activate_convproc(conu, dconudt, f_ent, dt_u, wup, tair,
                                         rhoair, kk, kactfirst);
          for (int i = 0; i < pcnst_extd; ++i)
            conu_dev[i] = conu[i];
          for (int i = 0; i < pcnst_extd; ++i)
            dconudt_dev[i] = dconudt[i];
        });
    // Check case of iflux_method == 2 which is not part of the e3sm tests.
    set_output(output, "conu", pcnst_extd, conu_host, conu_dev);
    set_output(output, "dconudt", pcnst_extd, dconudt_host, dconudt_dev);
  });
}
