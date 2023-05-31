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
} // namespace
void ma_precpevap(Ensemble *ensemble) {
  // We don't need any settings for this particular test.
  // Settings settings = ensemble->settings();
  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    const int nlev = 72;
    // Fetch ensemble parameters
    // Convert to C++ index by subtracting one.
    const int kk = input.get("kk") - 1;
    EKAT_ASSERT(53 == kk);
    const Real pr_flux = input.get("pr_flux");
    Real pr_flux_base = input.get("pr_flux_base");

    std::vector<Real> evapc_host, dpdry_i_host;
    ColumnView evapc_dev, dpdry_i_dev;
    get_input(input, "evapc", nlev, evapc_host, evapc_dev);
    get_input(input, "dpdry_i", nlev, dpdry_i_host, dpdry_i_dev);

    ColumnView return_vals = mam4::validation::create_column_view(3);
    Kokkos::parallel_for(
        "ma_precpevap", 1, KOKKOS_LAMBDA(int) {
          Real evapc[nlev];
          for (int i = 0; i < nlev; ++i)
            evapc[i] = evapc_dev(i);
          Real dpdry_i[nlev];
          for (int i = 0; i < nlev; ++i)
            dpdry_i[i] = dpdry_i_dev(i);
          Real pr_flux_tmp = -1, x_ratio = -1;
          Real flux_base = pr_flux_base;
          convproc::ma_precpevap(dpdry_i[kk], evapc[kk], pr_flux, flux_base,
                                 pr_flux_tmp, x_ratio);
          return_vals(0) = flux_base;
          return_vals(1) = pr_flux_tmp;
          return_vals(2) = x_ratio;
        });
    auto host_view = Kokkos::create_mirror_view(return_vals);
    Kokkos::deep_copy(host_view, return_vals);
    pr_flux_base = host_view(0);
    Real pr_flux_tmp = host_view(1);
    Real x_ratio = host_view(2);
    output.set("pr_flux_base", pr_flux_base);
    output.set("pr_flux_tmp", pr_flux_tmp);
    output.set("x_ratio", x_ratio);
  });

  // Check some corner cases.
  const int nlev = 72;
  ColumnView return_vals = mam4::validation::create_column_view(3);
  Kokkos::parallel_for(
      "ma_precpevap", 1, KOKKOS_LAMBDA(int) {
        const int kk = 1;
        Real evapc[nlev];
        for (int i = 0; i < nlev; ++i)
          evapc[i] = 99;
        Real dpdry_i[nlev];
        for (int i = 0; i < nlev; ++i)
          dpdry_i[i] = 1123;
        Real pr_flux = -1, pr_flux_tmp = -1, x_ratio = -1;
        Real flux_base = 1.0e-40;
        convproc::ma_precpevap(dpdry_i[kk], evapc[kk], pr_flux, flux_base,
                               pr_flux_tmp, x_ratio);
        return_vals(0) = flux_base;
        return_vals(1) = pr_flux_tmp;
        return_vals(2) = x_ratio;
      });
  auto host_view = Kokkos::create_mirror_view(return_vals);
  Kokkos::deep_copy(host_view, return_vals);
  Real pr_flux_base = host_view(0);
  Real pr_flux_tmp = host_view(1);
  Real x_ratio = host_view(2);
  EKAT_REQUIRE_MSG(pr_flux_base == 0, "Special case of zero input failed.");
  EKAT_REQUIRE_MSG(pr_flux_tmp == 0, "Special case of zero input failed.");
  EKAT_REQUIRE_MSG(0 == x_ratio, "Special case of zero input failed.");

  Kokkos::parallel_for(
      "ma_precpevap", 1, KOKKOS_LAMBDA(int) {
        const int kk = 1;
        Real evapc[nlev];
        for (int i = 0; i < nlev; ++i)
          evapc[i] = 99;
        Real dpdry_i[nlev];
        for (int i = 0; i < nlev; ++i)
          dpdry_i[i] = 1123;
        Real pr_flux = 0, pr_flux_tmp = -1, x_ratio = -1;
        Real flux_base = 1.0;
        convproc::ma_precpevap(dpdry_i[kk], evapc[kk], pr_flux, flux_base,
                               pr_flux_tmp, x_ratio);
        return_vals(0) = flux_base;
        return_vals(1) = pr_flux_tmp;
        return_vals(2) = x_ratio;
      });
  host_view = Kokkos::create_mirror_view(return_vals);
  Kokkos::deep_copy(host_view, return_vals);
  pr_flux_base = host_view(0);
  pr_flux_tmp = host_view(1);
  x_ratio = host_view(2);
  EKAT_REQUIRE_MSG(pr_flux_base == 0, "Special case of zero input failed.");
  EKAT_REQUIRE_MSG(pr_flux_tmp == 0, "Special case of zero input failed.");
  EKAT_REQUIRE_MSG(0 == x_ratio, "Special case of zero input failed.");
}
