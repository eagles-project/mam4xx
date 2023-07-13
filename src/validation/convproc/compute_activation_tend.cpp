// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

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
               Kokkos::View<Real **, Kokkos::MemoryUnmanaged> &dev) {
  host = input.get_array(name);
  ColumnView col_view = mam4::validation::create_column_view(rows * cols);
  dev = Kokkos::View<Real **, Kokkos::MemoryUnmanaged>(col_view.data(), rows,
                                                       cols);
  EKAT_ASSERT(host.size() == rows * cols);
  {
    std::vector<std::vector<Real>> matrix(rows, std::vector<Real>(cols));
    // Row Major layout
    for (int i = 0, n = 0; i < rows; ++i)
      for (int j = 0; j < cols; ++j, ++n)
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
                const Kokkos::View<Real **, Kokkos::MemoryUnmanaged> &dev) {
  auto host_view = Kokkos::create_mirror_view(dev);
  Kokkos::deep_copy(host_view, dev);
  for (int i = 0, n = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j, ++n)
      host[n] = host_view(i, j);
  output.set(name, host);
}
} // namespace
void compute_activation_tend(Ensemble *ensemble) {
  // We don't need any settings for this particular test.
  // Settings settings = ensemble->settings();
  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    const int nlev = 72;
    const int pcnst_extd = ConvProc::pcnst_extd;
    // Fetch ensemble parameters
    // Convert to C++ index by subtracting one.
    const int kk = input.get("kk") - 1;
    int kactcnt_host = input.get("kactcnt");
    int kactfirst_host = input.get("kactfirst") - 1;
    const Real f_ent = input.get("f_ent");
    Real xx_wcldbase_host = input.get("xx_wcldbase");
    int xx_kcldbase_host = input.get("xx_kcldbase");

    std::vector<Real> cldfrac_i_host, rhoair_i_host, mu_i_host, dt_u_host,
        wup_host, icwmr_host, temperature_host, conu_host, dconudt_activa_host;
    ColumnView cldfrac_i_dev, rhoair_i_dev, mu_i_dev, dt_u_dev, wup_dev,
        icwmr_dev, temperature_dev;

    Kokkos::View<Real **, Kokkos::MemoryUnmanaged> conu_dev, dconudt_activa_dev;

    get_input(input, "cldfrac_i", nlev, cldfrac_i_host, cldfrac_i_dev);
    get_input(input, "rhoair_i", nlev, rhoair_i_host, rhoair_i_dev);
    get_input(input, "mu_i", nlev + 1, mu_i_host, mu_i_dev);
    get_input(input, "dt_u", nlev, dt_u_host, dt_u_dev);
    get_input(input, "wup", nlev, wup_host, wup_dev);
    get_input(input, "icwmr", nlev, icwmr_host, icwmr_dev);
    get_input(input, "temperature", nlev, temperature_host, temperature_dev);
    get_input(input, "conu", nlev + 1, pcnst_extd, conu_host, conu_dev);
    get_input(input, "dconudt_activa", nlev + 1, pcnst_extd,
              dconudt_activa_host, dconudt_activa_dev);

    ColumnView scalars_dev = mam4::validation::create_column_view(4);
    Kokkos::parallel_for(
        "compute_activation_tend", 1, KOKKOS_LAMBDA(int) {
          Real cldfrac_i[nlev], rhoair_i[nlev], dt_u[nlev], wup[nlev],
              icwmr[nlev], temperature[nlev], mu_i[nlev + 1],
              conu[nlev + 1][pcnst_extd], dconudt_activa[nlev + 1][pcnst_extd];
          for (int i = 0; i < nlev; ++i) {
            cldfrac_i[i] = cldfrac_i_dev[i];
            rhoair_i[i] = rhoair_i_dev[i];
            dt_u[i] = dt_u_dev[i];
            wup[i] = wup_dev[i];
            icwmr[i] = icwmr_dev[i];
            temperature[i] = temperature_dev[i];
          }
          for (int i = 0; i < nlev + 1; ++i) {
            mu_i[i] = mu_i_dev[i];
            for (int j = 0; j < pcnst_extd; ++j) {
              conu[i][j] = conu_dev(i, j);
              dconudt_activa[i][j] = dconudt_activa_dev(i, j);
            }
          }
          int kactcnt = kactcnt_host;
          int kactfirst = kactfirst_host;
          Real xx_wcldbase = xx_wcldbase_host;
          int xx_kcldbase = xx_kcldbase_host;
          convproc::compute_activation_tend(
              f_ent, cldfrac_i[kk], rhoair_i[kk], mu_i[kk], mu_i[kk + 1],
              dt_u[kk], wup[kk], icwmr[kk], temperature[kk], kk, kactcnt,
              kactfirst, conu[kk], dconudt_activa[kk], xx_wcldbase,
              xx_kcldbase);

          for (int i = 0; i < nlev + 1; ++i) {
            for (int j = 0; j < pcnst_extd; ++j) {
              conu_dev(i, j) = conu[i][j];
              dconudt_activa_dev(i, j) = dconudt_activa[i][j];
            }
          }
          scalars_dev[0] = kactcnt;
          scalars_dev[1] = kactfirst;
          scalars_dev[2] = xx_wcldbase;
          scalars_dev[3] = xx_kcldbase;
        });
    set_output(output, "conu", nlev + 1, pcnst_extd, conu_host, conu_dev);
    set_output(output, "dconudt_activa", nlev + 1, pcnst_extd,
               dconudt_activa_host, dconudt_activa_dev);
    {
      auto host_view = Kokkos::create_mirror_view(scalars_dev);
      Kokkos::deep_copy(host_view, scalars_dev);
      kactcnt_host = host_view[0];
      kactfirst_host = host_view[1];
      xx_wcldbase_host = host_view[2];
      xx_kcldbase_host = host_view[3];
    }
    output.set("kactcnt", kactcnt_host);
    // Make indexes Fortran based again.
    output.set("kactfirst", 1 + kactfirst_host);
    output.set("xx_wcldbase", xx_wcldbase_host);
    output.set("xx_kcldbase", xx_kcldbase_host);
  });
}
