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

void init_scratch(
    Kokkos::View<Real *> scratch1Dviews[ConvProc::Col1DViewInd::NumScratch]) {
  const ConvProc::Config config_;
  Kokkos::resize(scratch1Dviews[ConvProc::q],
                 config_.nlev * ConvProc::gas_pcnst);
  Kokkos::resize(scratch1Dviews[ConvProc::mu], 1 + config_.nlev);
  Kokkos::resize(scratch1Dviews[ConvProc::md], 1 + config_.nlev);
  Kokkos::resize(scratch1Dviews[ConvProc::eudp], config_.nlev);
  Kokkos::resize(scratch1Dviews[ConvProc::dudp], config_.nlev);
  Kokkos::resize(scratch1Dviews[ConvProc::eddp], config_.nlev);
  Kokkos::resize(scratch1Dviews[ConvProc::dddp], config_.nlev);
  Kokkos::resize(scratch1Dviews[ConvProc::rhoair], config_.nlev);
  Kokkos::resize(scratch1Dviews[ConvProc::zmagl], config_.nlev);
  Kokkos::resize(scratch1Dviews[ConvProc::zmagl], config_.nlev);
  Kokkos::resize(scratch1Dviews[ConvProc::gath],
                 (1 + config_.nlev) * ConvProc::pcnst_extd);
  Kokkos::resize(scratch1Dviews[ConvProc::chat],
                 (1 + config_.nlev) * ConvProc::pcnst_extd);
  Kokkos::resize(scratch1Dviews[ConvProc::conu],
                 (1 + config_.nlev) * ConvProc::pcnst_extd);
  Kokkos::resize(scratch1Dviews[ConvProc::cond],
                 (1 + config_.nlev) * ConvProc::pcnst_extd);
  Kokkos::resize(scratch1Dviews[ConvProc::dconudt_wetdep],
                 (1 + config_.nlev) * ConvProc::pcnst_extd);
  Kokkos::resize(scratch1Dviews[ConvProc::dconudt_activa],
                 (1 + config_.nlev) * ConvProc::pcnst_extd);
  Kokkos::resize(scratch1Dviews[ConvProc::fa_u], config_.nlev);
  Kokkos::resize(scratch1Dviews[ConvProc::dcondt],
                 config_.nlev * ConvProc::pcnst_extd);
  Kokkos::resize(scratch1Dviews[ConvProc::dcondt_wetdep],
                 config_.nlev * ConvProc::pcnst_extd);
  Kokkos::resize(scratch1Dviews[ConvProc::dcondt_prevap],
                 config_.nlev * ConvProc::pcnst_extd);
  Kokkos::resize(scratch1Dviews[ConvProc::dcondt_prevap_hist],
                 config_.nlev * ConvProc::pcnst_extd);
  Kokkos::resize(scratch1Dviews[ConvProc::dcondt_resusp],
                 config_.nlev * ConvProc::pcnst_extd);
  Kokkos::resize(scratch1Dviews[ConvProc::wd_flux], ConvProc::pcnst_extd);
  Kokkos::resize(scratch1Dviews[ConvProc::sumactiva], ConvProc::pcnst_extd);
  Kokkos::resize(scratch1Dviews[ConvProc::sumaqchem], ConvProc::pcnst_extd);
  Kokkos::resize(scratch1Dviews[ConvProc::sumprevap], ConvProc::pcnst_extd);
  Kokkos::resize(scratch1Dviews[ConvProc::sumprevap_hist],
                 ConvProc::pcnst_extd);
  Kokkos::resize(scratch1Dviews[ConvProc::sumresusp], ConvProc::pcnst_extd);
  Kokkos::resize(scratch1Dviews[ConvProc::sumwetdep], ConvProc::pcnst_extd);
}

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
void get_input(
    const Input &input, const std::string &name, const int rows, const int cols,
    std::vector<Real> &host,
    Kokkos::View<Real * [ConvProc::gas_pcnst], Kokkos::MemoryUnmanaged> &dev) {
  host = input.get_array(name);
  ColumnView col_view = mam4::validation::create_column_view(rows * cols);
  dev = Kokkos::View<Real **, Kokkos::MemoryUnmanaged>(col_view.data(), rows,
                                                       cols);
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
template <int COLS>
void set_output(Output &output, const std::string &name, const int rows,
                const int cols, std::vector<Real> &host,
                const Kokkos::View<Real *[COLS]> &dev) {
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
void ma_convproc_dp_intr(Ensemble *ensemble) {
  // We don't need any settings for this particular test.
  // Settings settings = ensemble->settings();
  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    const int nlev = 72;
    const int ktop = 47;
    const int kbot = 71;
    const Real dt = 36000;
    // const int pcnst_extd = ConvProc::pcnst_extd;
    const int pcnst = ConvProc::gas_pcnst;
    const int nsrflx = 6;
    // Fetch ensemble parameters
    // Convert to C++ index by subtracting one.
    // ktop is jt(il1g) to jt(il2g) in Fortran
    // but jt is just a scalar and il1g=il2g=48;
    // kbot is mx(il1g) to jt(il2g) in Fortran
    // but mx is just a scalar iand l1g=il2g=71;
    EKAT_ASSERT(input.get("dt") == 3600);
    EKAT_ASSERT(input.get("nsrflx") == 6);
    EKAT_ASSERT(input.get("jt") == 72);
    EKAT_ASSERT(input.get("maxg") == 1);
    EKAT_ASSERT(input.get("ideep") == 0);
    EKAT_ASSERT(input.get("lengath") == 1);

    int mmtoo_prevap_resusp[pcnst];
    {
      const int resusp[pcnst] = {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                                 0,  0,  0,  0,  0,  31, 33, 34, 32, 29,
                                 30, 35, -2, 31, 34, 30, 35, -2, 29, 30,
                                 31, 32, 33, 34, 35, -2, 33, 32, 35, -2};
      for (int i = 0; i < pcnst; ++i)
        mmtoo_prevap_resusp[i] = resusp[i] - 1;
    }

    std::vector<Real> temperature_host, pmid_host, qnew_host, mu_host, md_host,
        du_host, eu_host, ed_host, dp_host, dpdry_host, cldfrac_host,
        icwmr_host, rprd_host, evapc_host, doconvproc_host, species_class_host,
        dqdt_host, qsrflx_host;
    ColumnView temperature_dev, pmid_dev, mu_dev, md_dev, du_dev, eu_dev,
        ed_dev, dp_dev, dpdry_dev, cldfrac_dev, icwmr_dev, rprd_dev, evapc_dev,
        doconvproc_dev, species_class_dev;

    Kokkos::View<Real *[pcnst]> dqdt_dev("dqdt", nlev, pcnst);
    Kokkos::View<bool *> dotend_dev("dotend", pcnst);
    Kokkos::View<Real *[nsrflx]> qsrflx_dev("qsrflx", pcnst, nsrflx);
    Kokkos::View<Real *[pcnst], Kokkos::MemoryUnmanaged> qnew_dev;

    get_input(input, "state_pdeldry", nlev, dpdry_host, dpdry_dev);
    get_input(input, "state_t", nlev, temperature_host, temperature_dev);
    get_input(input, "state_pmid", nlev, pmid_host, pmid_dev);
    get_input(input, "qnew", nlev, pcnst, qnew_host, qnew_dev);
    get_input(input, "dp_frac", nlev, cldfrac_host, cldfrac_dev);
    get_input(input, "icwmrdp", nlev, icwmr_host, icwmr_dev);
    get_input(input, "rprddp", nlev, rprd_host, rprd_dev);
    get_input(input, "evapcdp", nlev, evapc_host, evapc_dev);
    get_input(input, "mu", nlev, mu_host, mu_dev);
    get_input(input, "md", nlev, md_host, md_dev);
    get_input(input, "du", nlev, du_host, du_dev);
    get_input(input, "eu", nlev, eu_host, eu_dev);
    get_input(input, "ed", nlev, ed_host, ed_dev);
    get_input(input, "dp", nlev, dp_host, dp_dev);
    get_input(input, "species_class", pcnst, species_class_host,
              species_class_dev);

    // get_input(input, "doconvproc", pcnst, doconvproc_host, doconvproc_dev);
    //
    Kokkos::View<Real *> scratch1Dviews[ConvProc::Col1DViewInd::NumScratch];
    init_scratch(scratch1Dviews);

    Kokkos::parallel_for(
        "ma_convproc_dp_intr", 1, KOKKOS_LAMBDA(int) {
          Real cldfrac[nlev], icwmr[nlev], pmid[nlev], rprd[nlev], dpdry[nlev],
              evapc[nlev], du[nlev], eu[nlev], ed[nlev], dp[nlev],
              temperature[nlev], dqdt[nlev][pcnst], qsrflx[pcnst][nsrflx];
          int species_class[pcnst];
          bool dotend[pcnst];
          for (int i = 0; i < nlev; ++i) {
            cldfrac[i] = cldfrac_dev[i];
            icwmr[i] = icwmr_dev[i];
            temperature[i] = temperature_dev[i];
            pmid[i] = pmid_dev[i];
            rprd[i] = rprd_dev[i];
            dpdry[i] = dpdry_dev[i];
            evapc[i] = evapc_dev[i];
            du[i] = du_dev[i];
            eu[i] = eu_dev[i];
            ed[i] = ed_dev[i];
            dp[i] = dp_dev[i];
          }
          for (int i = 0; i < pcnst; ++i) {
            species_class[i] = species_class_dev[i];
          }
          auto dqdt_view = Kokkos::View<Real **, Kokkos::MemoryUnmanaged>(
              &dqdt[0][0], nlev, pcnst);
          convproc::ma_convproc_dp_intr(
              scratch1Dviews, nlev, temperature, pmid, dpdry, dt, cldfrac,
              icwmr, rprd, evapc, du, eu, ed, dp, ktop, kbot, qnew_dev,
              species_class, mmtoo_prevap_resusp, dqdt_view, qsrflx, dotend);

          for (int i = 0; i < nlev; ++i) {
            for (int j = 0; j < pcnst; ++j) {
              dqdt_dev(i, j) = dqdt_view(i, j);
            }
          }
          for (int i = 0; i < pcnst; ++i) {
            for (int j = 0; j < nsrflx; ++j) {
              qsrflx_dev(i, j) = qsrflx[i][j];
            }
          }
          for (int j = 0; j < pcnst; ++j) {
            dotend_dev(j) = dotend[j];
          }
        });
    set_output(output, "dqdt", nlev, pcnst, dqdt_host, dqdt_dev);
    set_output(output, "qsrflx", pcnst, nsrflx, qsrflx_host, qsrflx_dev);
    {
      std::vector<Real> host(pcnst);
      auto host_view = Kokkos::create_mirror_view(dotend_dev);
      Kokkos::deep_copy(host_view, dotend_dev);
      for (int i = 0; i < pcnst; ++i) {
        host[i] = host_view(i);
      }
      output.set("dotend", host);
    }
  });
}
