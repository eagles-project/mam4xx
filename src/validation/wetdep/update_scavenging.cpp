// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <iostream>
#include <mam4xx/mam4.hpp>
#include <mam4xx/wet_dep.hpp>
#include <skywalker.hpp>
#include <validation.hpp>
#include <vector>

using namespace haero;
using namespace skywalker;

void test_update_scavenging_process(const Input &input, Output &output) {
  // Ensemble parameters
  // Declare array of strings for input names
  std::string input_variables[] = {"dt",        "mam_prevap_resusp_optcc",
                                   "pdel_ik",   "omsm",
                                   "srcc",      "srcs",
                                   "srct",      "fins",
                                   "finc",      "fracev_st",
                                   "fracev_cu", "resusp_c",
                                   "resusp_s",  "precs_ik",
                                   "evaps_ik",  "cmfdqr_ik",
                                   "evapc_ik",  "scavabs",
                                   "scavabc",   "precabc",
                                   "precabs"};

  // Iterate over input_variables and error if not in input
  for (std::string name : input_variables) {
    if (!input.has(name.c_str())) {
      std::cerr << "Required name for variable: " << name << std::endl;
      exit(1);
    }
  }

  // Parse input
  // These first two values are unused
  EKAT_ASSERT(0 == input.get("dt"));
  const int mam_prevap_resusp_optcc = input.get("mam_prevap_resusp_optcc");
  const Real pdel_ik = input.get("pdel_ik");
  const Real omsm = input.get("omsm");
  const Real srcc = input.get("srcc");
  const Real srcs = input.get("srcs");
  const Real srct = input.get("srct");
  const Real fins = input.get("fins");
  const Real finc = input.get("finc");
  const Real fracev_st = input.get("fracev_st");
  const Real fracev_cu = input.get("fracev_cu");
  const Real resusp_c = input.get("resusp_c");
  const Real resusp_s = input.get("resusp_s");
  const Real precs_ik = input.get("precs_ik");
  const Real evaps_ik = input.get("evaps_ik");
  const Real cmfdqr_ik = input.get("cmfdqr_ik");
  const Real evapc_ik = input.get("evapc_ik");
  const Real scavabs = input.get("scavabs");
  const Real scavabc = input.get("scavabc");
  const Real precabc = input.get("precabc");
  const Real precabs = input.get("precabs");

  ColumnView return_vals = mam4::validation::create_column_view(12);
  Kokkos::parallel_for(
      "wetdep::update_scavenging", 1, KOKKOS_LAMBDA(const int) {
        Real scavt_ik, iscavt_ik, icscavt_ik, isscavt_ik, bcscavt_ik,
            bsscavt_ik, rcscavt_ik, rsscavt_ik,
            scavabs_d = scavabs, scavabc_d = scavabc, precabc_d = precabc,
            precabs_d = precabs;
        mam4::wetdep::update_scavenging(
            mam_prevap_resusp_optcc, pdel_ik, omsm, srcc, srcs, srct, fins,
            finc, fracev_st, fracev_cu, resusp_c, resusp_s, precs_ik, evaps_ik,
            cmfdqr_ik, evapc_ik, scavt_ik, iscavt_ik, icscavt_ik, isscavt_ik,
            bcscavt_ik, bsscavt_ik, rcscavt_ik, rsscavt_ik, scavabs_d,
            scavabc_d, precabc_d, precabs_d);

        return_vals[0] = scavt_ik;
        return_vals[1] = iscavt_ik;
        return_vals[2] = icscavt_ik;
        return_vals[3] = isscavt_ik;
        return_vals[4] = bcscavt_ik;
        return_vals[5] = bsscavt_ik;
        return_vals[6] = rcscavt_ik;
        return_vals[7] = rsscavt_ik;
        return_vals[8] = scavabs_d;
        return_vals[9] = scavabc_d;
        return_vals[10] = precabc_d;
        return_vals[11] = precabs_d;
      });

  // Create mirror views for output arrays
  auto vals_host = Kokkos::create_mirror_view(return_vals);

  // Copy values back to the host
  Kokkos::deep_copy(vals_host, return_vals);

  // Set the output values
  output.set("scavt_ik", vals_host[0]);
  output.set("iscavt_ik", vals_host[1]);
  output.set("icscavt_ik", vals_host[2]);
  output.set("isscavt_ik", vals_host[3]);
  output.set("bcscavt_ik", vals_host[4]);
  output.set("bsscavt_ik", vals_host[5]);
  output.set("rcscavt_ik", vals_host[6]);
  output.set("rsscavt_ik", vals_host[7]);
  output.set("scavabs", vals_host[8]);
  output.set("scavabc", vals_host[9]);
  output.set("precabc", vals_host[10]);
  output.set("precabs", vals_host[11]);
}

void test_update_scavenging(std::unique_ptr<Ensemble> &ensemble) {
  ensemble->process([&](const Input &input, Output &output) {
    test_update_scavenging_process(input, output);
  });
}
