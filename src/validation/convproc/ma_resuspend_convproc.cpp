// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/convproc.hpp>
#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;

namespace {
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
  auto host_view = Kokkos::create_mirror_view(dev);
  Kokkos::deep_copy(host_view, dev);
  for (int j = 0, n = 0; j < cols; ++j)
    for (int i = 0; i < rows; ++i, ++n)
      host[n] = host_view(i, j);
  output.set(name, host);
}
} // namespace
void ma_resuspend_convproc(Ensemble *ensemble) {

  // We don't need any settings for this particular test.
  // Settings settings = ensemble->settings();

  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    const int nlev = 72;
    const int gas_pcnst = ConvProc::gas_pcnst;
    // Fetch ensemble parameters

    // these variables depend on mode No and k
    const int ktop = input.get("ktop");
    const int kbot_prevap = input.get("kbot_prevap");
    EKAT_ASSERT(nlev == kbot_prevap);
    // number of tracers to transport
    const int pcnst_extd = input.get("pcnst_extd");
    EKAT_ASSERT(pcnst_extd == 2 * gas_pcnst);

    // flag for doing convective transport
    std::vector<Real> dcondt_host, dcondt_resusp_host;
    Kokkos::View<Real **, Kokkos::MemoryUnmanaged> dcondt_dev,
        dcondt_resusp_dev;
    get_input(input, "dcondt", pcnst_extd, nlev, dcondt_host, dcondt_dev);
    {
      dcondt_resusp_host.resize(nlev * pcnst_extd);
      ColumnView col_view =
          mam4::validation::create_column_view(nlev * pcnst_extd);
      dcondt_resusp_dev = Kokkos::View<Real **, Kokkos::MemoryUnmanaged>(
          col_view.data(), pcnst_extd, nlev);
    }
    Kokkos::parallel_for(
        "ma_resuspend_convproc", kbot_prevap, KOKKOS_LAMBDA(int klev) {
          if (ktop - 1 <= klev) {

            Real dcondt[2 * gas_pcnst];
            for (int i = 0; i < 2 * gas_pcnst; ++i)
              dcondt[i] = dcondt_dev(i, klev);
            Real dcondt_resusp[2 * gas_pcnst];
            convproc::ma_resuspend_convproc(dcondt, dcondt_resusp);

            for (int i = 0; i < 2 * gas_pcnst; ++i)
              dcondt_dev(i, klev) = dcondt[i];
            for (int i = 0; i < 2 * gas_pcnst; ++i)
              dcondt_resusp_dev(i, klev) = dcondt_resusp[i];
          }
        });
    set_output(output, "dcondt", pcnst_extd, nlev, dcondt_host, dcondt_dev);
    set_output(output, "dcondt_resusp", pcnst_extd, nlev, dcondt_resusp_host,
               dcondt_resusp_dev);
  });
}
