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
                const Kokkos::View<Real **, Kokkos::MemoryUnmanaged> &dev) {
  host.resize(rows * cols);
  auto host_view = Kokkos::create_mirror_view(dev);
  Kokkos::deep_copy(host_view, dev);
  for (int i = 0, n = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j, ++n)
      host[n] = host_view(i, j);
  output.set(name, host);
}
} // namespace
void initialize_tmr_array(Ensemble *ensemble) {
  // We don't need any settings for this particular test.
  // Settings settings = ensemble->settings();
  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    const int nlev = 72;
    // Fetch ensemble parameters
    // Convert to C++ index by subtracting one.
    const int pcnst_extd = input.get("pcnst_extd");
    EKAT_ASSERT(pcnst_extd == 80);
    const int ncnst = input.get("ncnst");
    EKAT_ASSERT(ncnst == 40);
    const int iconvtype = input.get("iconvtype");

    std::vector<Real> doconvproc_extd_host, q_i_host, gath_host, chat_host,
        conu_host, cond_host;
    ;
    ColumnView doconvproc_extd_dev;
    Kokkos::View<Real **, Kokkos::MemoryUnmanaged> q_i_dev, gath_dev, chat_dev,
        conu_dev, cond_dev;
    get_input(input, "doconvproc_extd", pcnst_extd, doconvproc_extd_host,
              doconvproc_extd_dev);
    get_input(input, "q_i", nlev, ncnst, q_i_host, q_i_dev);

    auto col_view = mam4::validation::create_column_view(pcnst_extd * nlev);
    gath_dev = Kokkos::View<Real **, Kokkos::MemoryUnmanaged>(col_view.data(),
                                                              nlev, pcnst_extd);
    col_view = mam4::validation::create_column_view(pcnst_extd * (nlev + 1));
    chat_dev = Kokkos::View<Real **, Kokkos::MemoryUnmanaged>(
        col_view.data(), nlev + 1, pcnst_extd);
    col_view = mam4::validation::create_column_view(pcnst_extd * (nlev + 1));
    conu_dev = Kokkos::View<Real **, Kokkos::MemoryUnmanaged>(
        col_view.data(), nlev + 1, pcnst_extd);
    col_view = mam4::validation::create_column_view(pcnst_extd * (nlev + 1));
    cond_dev = Kokkos::View<Real **, Kokkos::MemoryUnmanaged>(
        col_view.data(), nlev + 1, pcnst_extd);

    Kokkos::parallel_for(
        "initialize_tmr_array", 1, KOKKOS_LAMBDA(int) {
          const int pcnst_extd = ConvProc::pcnst_extd;
          const int pcnst = ConvProc::gas_pcnst;
          bool doconvproc_extd[pcnst_extd];
          for (int i = 0; i < pcnst_extd; ++i)
            doconvproc_extd[i] = doconvproc_extd_dev[i];
          Real q_i[nlev][pcnst];
          for (int i = 0; i < nlev; ++i)
            for (int j = 0; j < pcnst; ++j)
              q_i[i][j] = q_i_dev(i, j);
          Real gath[nlev][pcnst_extd];
          Real chat[nlev + 1][pcnst_extd];
          Real conu[nlev + 1][pcnst_extd];
          Real cond[nlev + 1][pcnst_extd];
          convproc::initialize_tmr_array(nlev, iconvtype, doconvproc_extd, q_i,
                                         gath, chat, conu, cond);
          for (int i = 0; i < nlev; ++i)
            for (int j = 0; j < pcnst_extd; ++j)
              gath_dev(i, j) = gath[i][j];
          for (int i = 0; i < nlev + 1; ++i)
            for (int j = 0; j < pcnst_extd; ++j)
              chat_dev(i, j) = chat[i][j];
          for (int i = 0; i < nlev + 1; ++i)
            for (int j = 0; j < pcnst_extd; ++j)
              conu_dev(i, j) = conu[i][j];
          for (int i = 0; i < nlev + 1; ++i)
            for (int j = 0; j < pcnst_extd; ++j)
              cond_dev(i, j) = cond[i][j];
        });
    // Check case of iflux_method == 2 which is not part of the e3sm tests.
    set_output(output, "const", nlev, pcnst_extd, gath_host, gath_dev);
    set_output(output, "chat", nlev + 1, pcnst_extd, chat_host, chat_dev);
    set_output(output, "conu", nlev + 1, pcnst_extd, conu_host, conu_dev);
    set_output(output, "cond", nlev + 1, pcnst_extd, cond_host, cond_dev);
  });
}
