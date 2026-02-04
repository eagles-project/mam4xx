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
        Real scavt = 0, iscavt = 0, icscavt = 0, isscavt = 0, bcscavt = 0,
             bsscavt = 0, rcscavt = 0, rsscavt = 0, scavabs_d = scavabs,
             scavabc_d = scavabc, precabc_d = precabc, precabs_d = precabs;

        // mam4::wetdep::update_scavenging(
        // mam_prevap_resusp_optcc, pdel, omsm, srcc, srcs, srct, fins,
        // finc, fracev_st, fracev_cu, resusp_c, resusp_s, precs, evaps,
        // cmfdqr, evapc, scavt, bcscavt, rcscavt, rsscavt,
        // scavabs_d, scavabc_d);

        const Real gravit = Constants::gravity;

        if (mam_prevap_resusp_optcc == 0)
          scavt = -srct + (fracev_st * scavabs + fracev_cu * scavabc) * gravit /
                              pdel_ik;
        else
          scavt = -srct + (resusp_s + resusp_c) * gravit / pdel_ik;

        if (mam_prevap_resusp_optcc == 0) {
          bcscavt = -(srcc * (1 - finc)) * omsm +
                    fracev_cu * scavabc * gravit / pdel_ik;
          rcscavt = 0.0;
          rsscavt = 0.0;
        } else {
          bcscavt = -(srcc * (1 - finc)) * omsm;
          rcscavt = resusp_c * gravit / pdel_ik;
          rsscavt = resusp_s * gravit / pdel_ik;
        }

        return_vals[0] = scavt;
        return_vals[1] = iscavt;
        return_vals[2] = icscavt;
        return_vals[3] = isscavt;
        return_vals[4] = bcscavt;
        return_vals[5] = bsscavt;
        return_vals[6] = rcscavt;
        return_vals[7] = rsscavt;
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
