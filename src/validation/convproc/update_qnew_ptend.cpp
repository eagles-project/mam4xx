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

void set_output(Output &output, const std::string &name, const int size,
                std::vector<Real> &host, const ColumnView &dev) {
  auto host_view = Kokkos::create_mirror_view(dev);
  Kokkos::deep_copy(host_view, dev);
  for (int n = 0; n < size; ++n)
    host[n] = host_view[n];
  output.set(name, host);
}
} // namespace
void update_qnew_ptend(Ensemble *ensemble) {

  // We don't need any settings for this particular test.
  // Settings settings = ensemble->settings();

  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    const int gas_pcnst = ConvProc::gas_pcnst;
    // Fetch ensemble parameters

    // delta t (model time increment) [s]
    const Real dt = input.get("dt");
    EKAT_ASSERT(dt == 3600);
    // flag for doing convective transport
    const bool is_update_ptend = input.get("is_update_ptend");

    std::vector<Real> dotend_host, dqdt_host, ptend_lq_host, ptend_q_host,
        qnew_host;
    ColumnView dotend_dev, ptend_lq_dev, dqdt_dev, ptend_q_dev, qnew_dev;

    get_input(input, "dotend", gas_pcnst, dotend_host, dotend_dev);
    get_input(input, "dqdt", gas_pcnst, dqdt_host, dqdt_dev);
    get_input(input, "ptend_lq", gas_pcnst, ptend_lq_host, ptend_lq_dev);
    get_input(input, "ptend_q", gas_pcnst, ptend_q_host, ptend_q_dev);
    get_input(input, "qnew", gas_pcnst, qnew_host, qnew_dev);

    Kokkos::parallel_for(
        "update_qnew_ptend", 1, KOKKOS_LAMBDA(int) {
          bool dotend[gas_pcnst];
          for (int n = 0; n < gas_pcnst; ++n)
            dotend[n] = dotend_dev[n];
          bool ptend_lq[gas_pcnst];
          for (int n = 0; n < gas_pcnst; ++n)
            ptend_lq[n] = ptend_lq_dev[n];

          Real dqdt[gas_pcnst];
          for (int j = 0; j < gas_pcnst; ++j)
            dqdt[j] = dqdt_dev(j);
          Real ptend_q[gas_pcnst];
          for (int j = 0; j < gas_pcnst; ++j)
            ptend_q[j] = ptend_q_dev(j);
          Real qnew[gas_pcnst];
          for (int j = 0; j < gas_pcnst; ++j)
            qnew[j] = qnew_dev(j);

          convproc::update_qnew_ptend(dotend, is_update_ptend, dqdt, dt,
                                      ptend_lq, ptend_q, qnew);
          for (int n = 0; n < gas_pcnst; ++n)
            ptend_lq_dev[n] = ptend_lq[n];
          for (int j = 0; j < gas_pcnst; ++j)
            ptend_q_dev(j) = ptend_q[j];
          for (int j = 0; j < gas_pcnst; ++j)
            qnew_dev(j) = qnew[j];
        });

    set_output(output, "ptend_lq", gas_pcnst, ptend_lq_host, ptend_lq_dev);
    set_output(output, "ptend_q", gas_pcnst, ptend_q_host, ptend_q_dev);
    set_output(output, "qnew", gas_pcnst, qnew_host, qnew_dev);
  });
}
