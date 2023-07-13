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
void compute_wup(Ensemble *ensemble) {
  // We don't need any settings for this particular test.
  // Settings settings = ensemble->settings();
  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    const int nlev = 72;
    // Fetch ensemble parameters
    // Convert to C++ index by subtracting one.
    const int kk = input.get("kk") - 1;
    EKAT_ASSERT(kk == 53);
    const Real iconvtype = input.get("iconvtype");
    EKAT_ASSERT(iconvtype == 1);

    std::vector<Real> mu_i_host, cldfrac_i_host, rhoair_i_host, zmagl_host,
        wup_host, wup_2_host, wup_3_host;
    ColumnView mu_i_dev, cldfrac_i_dev, rhoair_i_dev, zmagl_dev, wup_dev,
        wup_2_dev, wup_3_dev;
    get_input(input, "mu_i", nlev + 1, mu_i_host, mu_i_dev);
    get_input(input, "cldfrac_i", nlev, cldfrac_i_host, cldfrac_i_dev);
    get_input(input, "rhoair_i", nlev, rhoair_i_host, rhoair_i_dev);
    get_input(input, "zmagl", nlev, zmagl_host, zmagl_dev);
    get_input(input, "wup", nlev, wup_host, wup_dev);
    wup_2_dev = mam4::validation::create_column_view(nlev);
    wup_3_dev = mam4::validation::create_column_view(nlev);
    Kokkos::parallel_for(
        "compute_wup", 1, KOKKOS_LAMBDA(int) {
          Real mu_i[nlev + 1];
          for (int i = 0; i < nlev + 1; ++i)
            mu_i[i] = mu_i_dev[i];
          Real cldfrac_i[nlev];
          for (int i = 0; i < nlev; ++i)
            cldfrac_i[i] = cldfrac_i_dev[i];
          Real rhoair_i[nlev];
          for (int i = 0; i < nlev; ++i)
            rhoair_i[i] = rhoair_i_dev[i];
          Real zmagl[nlev];
          for (int i = 0; i < nlev; ++i)
            zmagl[i] = zmagl_dev[i];
          Real zmagl_2[nlev];
          for (int i = 0; i < nlev; ++i)
            zmagl_2[i] = zmagl_dev[i];
          Real wup[nlev];
          for (int i = 0; i < nlev; ++i)
            wup[i] = wup_dev[i];
          Real wup_2[nlev];
          for (int i = 0; i < nlev; ++i)
            wup_2[i] = wup_dev[i];
          Real wup_3[nlev];
          for (int i = 0; i < nlev; ++i)
            wup_3[i] = wup_dev[i];

          wup[kk] =
              convproc::compute_wup(iconvtype, mu_i[kk], mu_i[kk + 1],
                                    cldfrac_i[kk], rhoair_i[kk], zmagl[kk]);
          const int iconvtype_2 = 2;
          wup_2[kk] =
              convproc::compute_wup(iconvtype_2, mu_i[kk], mu_i[kk + 1],
                                    cldfrac_i[kk], rhoair_i[kk], zmagl[kk]);

          zmagl_2[kk] /= 10;
          wup_3[kk] =
              convproc::compute_wup(iconvtype, mu_i[kk], mu_i[kk + 1],
                                    cldfrac_i[kk], rhoair_i[kk], zmagl_2[kk]);
          for (int i = 0; i < nlev; ++i)
            wup_dev[i] = wup[i];
          for (int i = 0; i < nlev; ++i)
            wup_2_dev[i] = wup_2[i];
          for (int i = 0; i < nlev; ++i)
            wup_3_dev[i] = wup_3[i];
        });
    // This special test checks an if statement in compute_wup that
    // sets hygro to 0.2 in case of very small volume. It is tripped twice:
    wup_2_host.resize(nlev);
    {
      auto host_view = Kokkos::create_mirror_view(wup_2_dev);
      Kokkos::deep_copy(host_view, wup_2_dev);
      for (int n = 0; n < nlev; ++n)
        wup_2_host[n] = host_view[n];
    }
    const Real wup_2_chk[nlev] = {0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0.981255604817,
                                  3.42602978,
                                  3.341147389,
                                  3.266199917,
                                  3.198549306,
                                  3.131584108,
                                  3.061526817,
                                  2.985477952,
                                  2.803891447,
                                  2.618128155,
                                  2.427279426,
                                  2.23011197,
                                  2.02485641,
                                  1.80882474,
                                  1.577618489,
                                  1.323448376,
                                  1.029998341,
                                  0.6919802548,
                                  0};
    for (int i = 0; i < nlev; ++i)
      EKAT_ASSERT(std::abs(wup_2_host[i] - wup_2_chk[i]) < .000001);

    set_output(output, "wup", nlev, wup_host, wup_dev);

    wup_3_host.resize(nlev);
    {
      auto host_view = Kokkos::create_mirror_view(wup_3_dev);
      Kokkos::deep_copy(host_view, wup_3_dev);
      for (int n = 0; n < nlev; ++n)
        wup_3_host[n] = host_view[n];
    }
    for (int i = 0; i < nlev; ++i) {
      if (53 == i)
        EKAT_ASSERT(std::abs(wup_3_host[i] - 1.39358046842) < .000001);
      else
        EKAT_ASSERT(std::abs(wup_3_host[i] - wup_host[i]) < .000001);
    }
  });
}
