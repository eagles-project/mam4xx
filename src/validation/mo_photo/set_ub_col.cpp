// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/mam4.hpp>

#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;
using namespace haero;
void set_ub_col(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    using View1D = typename DeviceType::view_1d<Real>;
    using View2D = typename DeviceType::view_2d<Real>;

    constexpr int pver = mam4::nlev;
    constexpr int gas_pcnst = mam4::gas_chemistry::gas_pcnst;
    constexpr int nfs = mam4::gas_chemistry::nfs;

    const Real spc_exo_col = input.get_array("spc_exo_col")[0];
    const auto vmr_in = input.get_array("vmr");
    const auto invariants_in = input.get_array("invariants");
    const auto pdel_in = input.get_array("pdel");

    View2D vmr("vmr", pver, gas_pcnst);
    View2D invariants("invariants", pver, nfs);
    View1D pdel("pdel", pver);

    mam4::validation::convert_1d_vector_to_2d_view_device(vmr_in, vmr);
    mam4::validation::convert_1d_vector_to_2d_view_device(invariants_in,
                                                          invariants);
    auto pdel_host = Kokkos::create_mirror_view(pdel);
    std::copy(pdel_in.data(), pdel_in.data() + pver, pdel_host.data());
    Kokkos::deep_copy(pdel, pdel_host);

    View1D col_delta("col_delta", pver + 1);
    col_delta(0) = spc_exo_col;
    Kokkos::parallel_for(
        pver, KOKKOS_LAMBDA(const int k) {
          Real vmr_ik[gas_pcnst];
          for (int l = 0; l < gas_pcnst; ++l) {
            vmr_ik[l] = vmr(k, l);
          }
          Real inv_ik[nfs];
          for (int l = 0; l < nfs; ++l) {
            inv_ik[l] = invariants(k, l);
          }
          mam4::mo_photo::set_ub_col(col_delta(k + 1), vmr_ik, inv_ik, pdel(k));
        });

    auto col_delta_host = Kokkos::create_mirror_view(col_delta);
    std::vector<Real> col_delta_out(col_delta_host.data(),
                                    col_delta_host.data() + pver + 1);
    output.set("col_delta", col_delta_out);
  });
}
