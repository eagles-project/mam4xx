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
  output.set(name, host);
}
} // namespace
void compute_downdraft_mixing_ratio(Ensemble *ensemble) {
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

    std::vector<Real> doconvproc_extd_host, md_i_host, eddp_host, gath_host,
        cond_host;
    ColumnView doconvproc_extd_dev, md_i_dev, eddp_dev;
    Kokkos::View<Real **, Kokkos::MemoryUnmanaged> gath_dev, cond_dev;

    get_input(input, "doconvproc_extd", ConvProc::pcnst_extd,
              doconvproc_extd_host, doconvproc_extd_dev);
    get_input(input, "md_i", nlev + 1, md_i_host, md_i_dev);
    get_input(input, "eddp", nlev, eddp_host, eddp_dev);
    get_input(input, "const", nlev, ConvProc::pcnst_extd, gath_host, gath_dev);
    get_input(input, "cond", nlev + 1, ConvProc::pcnst_extd, cond_host,
              cond_dev);

    Kokkos::parallel_for(
        "compute_downdraft_mixing_ratio", 1, KOKKOS_LAMBDA(int) {
          bool doconvproc_extd[ConvProc::pcnst_extd];
          Real md_i[nlev + 1];
          Real eddp[nlev];
          Real gath[nlev][ConvProc::pcnst_extd];
          Real cond[nlev + 1][ConvProc::pcnst_extd];

          for (int i = 0; i < ConvProc::pcnst_extd; ++i)
            doconvproc_extd[i] = doconvproc_extd_dev[i];
          for (int i = 0; i < nlev + 1; ++i)
            md_i[i] = md_i_dev(i);
          for (int i = 0; i < nlev; ++i)
            eddp[i] = eddp_dev(i);
          for (int i = 0; i < nlev; ++i)
            for (int j = 0; j < ConvProc::pcnst_extd; ++j)
              gath[i][j] = gath_dev(i, j);
          for (int i = 0; i < nlev + 1; ++i)
            for (int j = 0; j < ConvProc::pcnst_extd; ++j)
              cond[i][j] = cond_dev(i, j);

          convproc::compute_downdraft_mixing_ratio(doconvproc_extd, ktop, kbot,
                                                   md_i, eddp, gath, cond);

          for (int i = 0; i < nlev + 1; ++i)
            for (int j = 0; j < ConvProc::pcnst_extd; ++j)
              cond_dev(i, j) = cond[i][j];
        });
    set_output(output, "cond", nlev + 1, ConvProc::pcnst_extd, cond_host,
               cond_dev);
  });
}
