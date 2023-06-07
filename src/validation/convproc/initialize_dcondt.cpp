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
  EKAT_ASSERT(host.size() == size);
  dev = mam4::validation::create_column_view(size);
  auto host_view = Kokkos::create_mirror_view(dev);
  for (int n = 0; n < size; ++n)
    host_view[n] = host[n];
  Kokkos::deep_copy(dev, host_view);
}
void get_input(const Input &input, const std::string &name, const int rows,
               const int cols, std::vector<Real> &host,
               Kokkos::View<Real **, Kokkos::MemoryUnmanaged> &dev) {
  host = input.get_array(name);
  EKAT_ASSERT(host.size() == rows * cols);
  ColumnView col_view = mam4::validation::create_column_view(rows * cols);
  dev = Kokkos::View<Real **, Kokkos::MemoryUnmanaged>(col_view.data(), rows,
                                                       cols);
  {
    std::vector<std::vector<Real>> matrix(rows, std::vector<Real>(cols));
    // Col Major layout
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
  host.resize(rows * cols);
  auto host_view = Kokkos::create_mirror_view(dev);
  Kokkos::deep_copy(host_view, dev);
  for (int i = 0, n = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j, ++n)
      host[n] = host_view(i, j);

  std::cout << __FILE__ << ":" << __LINE__ << " dcondt[" << 48 << "][" << 62
            << "]:" << host_view(48, 62) << std::endl;
  output.set(name, host);
}
} // namespace
void initialize_dcondt(Ensemble *ensemble) {
  // We don't need any settings for this particular test.
  // Settings settings = ensemble->settings();
  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    const int nlev = 72;
    // Fetch ensemble parameters
    // Convert to C++ index by subtracting one.
    const int ktop = input.get("ktop") - 1;
    EKAT_ASSERT(47 == ktop);
    const int kbot = input.get("kbot");
    EKAT_ASSERT(71 == kbot);
    const int iflux_method = input.get("iflux_method");
    EKAT_ASSERT(1 == iflux_method);

    std::vector<Real> doconvproc_extd_host, dpdry_i_host, fa_u_host, mu_i_host,
        md_i_host, chat_host, gath_host, conu_host, cond_host,
        dconudt_activa_host, dconudt_wetdep_host, dudp_host, dddp_host,
        eudp_host, eddp_host, dcondt_host;
    ColumnView doconvproc_extd_dev, dpdry_i_dev, fa_u_dev, mu_i_dev, md_i_dev,
        dudp_dev, dddp_dev, eudp_dev, eddp_dev;
    Kokkos::View<Real **, Kokkos::MemoryUnmanaged> gath_dev, chat_dev, conu_dev,
        cond_dev, dconudt_activa_dev, dconudt_wetdep_dev, dcondt_dev;

    get_input(input, "doconvproc_extd", ConvProc::pcnst_extd,
              doconvproc_extd_host, doconvproc_extd_dev);

    get_input(input, "dpdry_i", nlev, dpdry_i_host, dpdry_i_dev);
    get_input(input, "fa_u", nlev, fa_u_host, fa_u_dev);
    get_input(input, "dudp", nlev, dudp_host, dudp_dev);
    get_input(input, "dddp", nlev, dddp_host, dddp_dev);
    get_input(input, "eudp", nlev, eudp_host, eudp_dev);
    get_input(input, "eddp", nlev, eddp_host, eddp_dev);

    get_input(input, "mu_i", nlev + 1, mu_i_host, mu_i_dev);
    get_input(input, "md_i", nlev + 1, md_i_host, md_i_dev);

    get_input(input, "chat", nlev + 1, ConvProc::pcnst_extd, chat_host,
              chat_dev);
    get_input(input, "const", nlev, ConvProc::pcnst_extd, gath_host, gath_dev);
    get_input(input, "conu", nlev + 1, ConvProc::pcnst_extd, conu_host,
              conu_dev);
    get_input(input, "cond", nlev + 1, ConvProc::pcnst_extd, cond_host,
              cond_dev);

    get_input(input, "dconudt_activa", nlev + 1, ConvProc::pcnst_extd,
              dconudt_activa_host, dconudt_activa_dev);
    get_input(input, "dconudt_wetdep", nlev + 1, ConvProc::pcnst_extd,
              dconudt_wetdep_host, dconudt_wetdep_dev);

    ColumnView col_view =
        mam4::validation::create_column_view(nlev * ConvProc::pcnst_extd);
    dcondt_dev = Kokkos::View<Real **, Kokkos::MemoryUnmanaged>(
        col_view.data(), nlev, ConvProc::pcnst_extd);

    Kokkos::parallel_for(
        "initialize_dcondt", 1, KOKKOS_LAMBDA(int) {
          bool doconvproc_extd[ConvProc::pcnst_extd];
          Real dpdry_i[nlev];
          Real fa_u[nlev];
          Real mu_i[nlev + 1];
          Real md_i[nlev + 1];
          Real chat[nlev + 1][ConvProc::pcnst_extd];
          Real gath[nlev][ConvProc::pcnst_extd];
          Real conu[nlev + 1][ConvProc::pcnst_extd];
          Real cond[nlev + 1][ConvProc::pcnst_extd];
          Real dconudt_activa[nlev + 1][ConvProc::pcnst_extd];
          Real dconudt_wetdep[nlev + 1][ConvProc::pcnst_extd];
          Real dudp[nlev];
          Real dddp[nlev];
          Real eudp[nlev];
          Real eddp[nlev];
          Real dcondt[nlev][ConvProc::pcnst_extd];

          for (int i = 0; i < ConvProc::pcnst_extd; ++i)
            doconvproc_extd[i] = doconvproc_extd_dev[i];
          for (int i = 0; i < nlev; ++i)
            dpdry_i[i] = dpdry_i_dev(i);
          for (int i = 0; i < nlev; ++i)
            fa_u[i] = fa_u_dev(i);
          for (int i = 0; i < nlev + 1; ++i)
            mu_i[i] = mu_i_dev(i);
          for (int i = 0; i < nlev + 1; ++i)
            md_i[i] = md_i_dev(i);
          for (int i = 0; i < nlev + 1; ++i)
            for (int j = 0; j < ConvProc::pcnst_extd; ++j)
              chat[i][j] = chat_dev(i, j);
          for (int i = 0; i < nlev; ++i)
            for (int j = 0; j < ConvProc::pcnst_extd; ++j)
              gath[i][j] = gath_dev(i, j);
          for (int i = 0; i < nlev + 1; ++i)
            for (int j = 0; j < ConvProc::pcnst_extd; ++j)
              conu[i][j] = conu_dev(i, j);
          for (int i = 0; i < nlev + 1; ++i)
            for (int j = 0; j < ConvProc::pcnst_extd; ++j)
              cond[i][j] = cond_dev(i, j);
          for (int i = 0; i < nlev + 1; ++i)
            for (int j = 0; j < ConvProc::pcnst_extd; ++j)
              dconudt_activa[i][j] = dconudt_activa_dev(i, j);
          for (int i = 0; i < nlev + 1; ++i)
            for (int j = 0; j < ConvProc::pcnst_extd; ++j)
              dconudt_wetdep[i][j] = dconudt_wetdep_dev(i, j);
          for (int i = 0; i < nlev; ++i)
            dudp[i] = dudp_dev(i);
          for (int i = 0; i < nlev; ++i)
            dddp[i] = dddp_dev(i);
          for (int i = 0; i < nlev; ++i)
            eudp[i] = eudp_dev(i);
          for (int i = 0; i < nlev; ++i)
            eddp[i] = eddp_dev(i);

          convproc::initialize_dcondt(
              doconvproc_extd, iflux_method, ktop, kbot, nlev, dpdry_i, fa_u,
              mu_i, md_i, chat, gath, conu, cond, dconudt_activa,
              dconudt_wetdep, dudp, dddp, eudp, eddp, dcondt);
          for (int i = 0; i < nlev; ++i)
            for (int j = 0; j < ConvProc::pcnst_extd; ++j)
              dcondt_dev(i, j) = dcondt[i][j];
        });
    set_output(output, "dcondt", nlev, ConvProc::pcnst_extd, dcondt_host,
               dcondt_dev);
  });

  // Check some corner cases.
  const int nlev = 72;
  ColumnView return_vals = mam4::validation::create_column_view(6);
  ColumnView dcondt_wetdep =
      mam4::validation::create_column_view(ConvProc::pcnst_extd);
  ColumnView dcondt =
      mam4::validation::create_column_view(ConvProc::pcnst_extd);
  ColumnView dcondt_prevap =
      mam4::validation::create_column_view(ConvProc::pcnst_extd);
  ColumnView dcondt_prevap_hist =
      mam4::validation::create_column_view(ConvProc::pcnst_extd);
  ColumnView wd_flux =
      mam4::validation::create_column_view(ConvProc::pcnst_extd);
  Kokkos::parallel_for(
      "initialize_dcondt", 1, KOKKOS_LAMBDA(int) {
        Real rprd[nlev];
        for (int i = 0; i < nlev; ++i)
          rprd[i] = 1;
        Real dpdry_i[nlev];
        for (int i = 0; i < nlev; ++i)
          dpdry_i[i] = 1;
        bool doconvproc_extd[ConvProc::pcnst_extd];
        for (int i = 0; i < ConvProc::pcnst_extd; ++i)
          doconvproc_extd[i] = 1;
        int species_class[ConvProc::gas_pcnst];
        for (int i = 0; i < ConvProc::gas_pcnst; ++i)
          species_class[i] = 1;
        int mmtoo_prevap_resusp[ConvProc::gas_pcnst];
        for (int i = 0; i < ConvProc::gas_pcnst; ++i)
          mmtoo_prevap_resusp[i] = 1;

        for (int i = 0; i < ConvProc::pcnst_extd; ++i)
          wd_flux[i] = 1;
        for (int i = 0; i < ConvProc::pcnst_extd; ++i)
          dcondt_wetdep[i] = 1;
        for (int i = 0; i < ConvProc::pcnst_extd; ++i)
          dcondt[i] = 1;
        for (int i = 0; i < ConvProc::pcnst_extd; ++i)
          dcondt_prevap[i] = 1;
        for (int i = 0; i < ConvProc::pcnst_extd; ++i)
          dcondt_prevap_hist[i] = 1;
        Real flux = 1;
        Real flux_tmp = 0;
        Real flux_base = 0;
        Real x_ratio = 0;
        int kk = 1;
        convproc::ma_precpprod(rprd[kk], dpdry_i[kk], doconvproc_extd, x_ratio,
                               species_class, mmtoo_prevap_resusp, flux,
                               flux_tmp, flux_base, wd_flux, dcondt_wetdep,
                               dcondt, dcondt_prevap, dcondt_prevap_hist);
        return_vals[0] = flux;
        return_vals[1] = flux_tmp;
        return_vals[2] = flux_base;
        return_vals[3] = 0;
        return_vals[4] = 0;
        return_vals[5] = 0;
        for (int i = 0; i < ConvProc::pcnst_extd; ++i)
          return_vals[3] += dcondt[i];
        for (int i = 0; i < ConvProc::pcnst_extd; ++i)
          return_vals[4] += dcondt_prevap[i];
        for (int i = 0; i < ConvProc::pcnst_extd; ++i)
          return_vals[5] += dcondt_prevap_hist[i];
      });
  auto host_view = Kokkos::create_mirror_view(return_vals);
  Kokkos::deep_copy(host_view, return_vals);
  Real pr_flux_base = host_view(0);
  Real pr_flux_tmp = host_view(1);
  Real x_ratio = host_view(2);
  Real x_dcondt = host_view(3);
  Real x_dcondt_prevap = host_view(4);
  Real x_dcondt_prevap_hist = host_view(5);
  EKAT_REQUIRE_MSG(pr_flux_base == 1, "Special case of input failed.");
  EKAT_REQUIRE_MSG(pr_flux_tmp == 0, "Special case of input failed.");
  EKAT_REQUIRE_MSG(x_ratio == 1, "Special case of input failed.");
  EKAT_REQUIRE_MSG(x_dcondt == 1 + 2 * (ConvProc::pcnst_extd - 1),
                   "Special case of input failed.");
  EKAT_REQUIRE_MSG(x_dcondt_prevap == 1 + 2 * (ConvProc::pcnst_extd - 1),
                   "Special case of input failed.");
  EKAT_REQUIRE_MSG(x_dcondt_prevap_hist == 1 + 2 * (ConvProc::pcnst_extd - 1),
                   "Special case of input failed.");
}
