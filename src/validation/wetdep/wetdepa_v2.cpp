// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/mam4.hpp>

#include <validation.hpp>
#include <vector>

using namespace skywalker;

void test_wetdepa_v2_process(const Input &input, Output &output) {
  const int nlev = 72;
  // Ensemble parameters
  // Declare array of strings for input names
  std::string input_variables[] = {"dt",
                                   "ncol",
                                   "deltat",
                                   "pdel",
                                   "cmfdqr",
                                   "evapc",
                                   "dlf",
                                   "conicw",
                                   "precs",
                                   "evaps",
                                   "cwat",
                                   "cldt",
                                   "cldc",
                                   "cldvcu",
                                   "cldvst",
                                   "tracer",
                                   "mam_prevap_resusp_optcc",
                                   "is_strat_cloudborne",
                                   "scavcoef",
                                   "f_act_conv",
                                   "qqcw",
                                   "sol_factb",
                                   "sol_facti",
                                   "sol_factic"};

  // Iterate over input_variables and error if not in input
  for (std::string name : input_variables) {
    if (!input.has(name.c_str()) && !input.has_array(name.c_str())) {
      std::cerr << "Required name for variable: " << name << std::endl;
      exit(1);
    }
  }

  // Parse input
  // These first two values are unused
  EKAT_ASSERT(0 == input.get("dt"));
  EKAT_ASSERT(4 == input.get("ncol"));
  const int is_strat_cloudborne = input.get("is_strat_cloudborne");
  const int mam_prevap_resusp_optcc = input.get("mam_prevap_resusp_optcc");
  const Real deltat = input.get("deltat");
  const Real sol_factb = input.get("sol_factb");
  const Real sol_facti = input.get("sol_facti");

  const std::vector<Real> pdel = input.get_array("pdel");
  const std::vector<Real> cmfdqr = input.get_array("cmfdqr");
  const std::vector<Real> evapc = input.get_array("evapc");
  const std::vector<Real> dlf = input.get_array("dlf");
  const std::vector<Real> conicw = input.get_array("conicw");
  const std::vector<Real> precs = input.get_array("precs");
  const std::vector<Real> evaps = input.get_array("evaps");
  const std::vector<Real> cwat = input.get_array("cwat");
  const std::vector<Real> cldt = input.get_array("cldt");
  const std::vector<Real> cldc = input.get_array("cldc");
  const std::vector<Real> cldvcu = input.get_array("cldvcu");
  const std::vector<Real> cldvst = input.get_array("cldvst");
  const std::vector<Real> tracer = input.get_array("tracer");
  const std::vector<Real> scavcoef = input.get_array("scavcoef");
  const std::vector<Real> f_act_conv = input.get_array("f_act_conv");
  const std::vector<Real> qqcw = input.get_array("qqcw");
  const std::vector<Real> sol_factic = input.get_array("sol_factic");

  auto pdel_dev = mam4::validation::create_column_view(nlev);
  auto cmfdqr_dev = mam4::validation::create_column_view(nlev);
  auto evapc_dev = mam4::validation::create_column_view(nlev);
  auto dlf_dev = mam4::validation::create_column_view(nlev);
  auto conicw_dev = mam4::validation::create_column_view(nlev);
  auto precs_dev = mam4::validation::create_column_view(nlev);
  auto evaps_dev = mam4::validation::create_column_view(nlev);
  auto cwat_dev = mam4::validation::create_column_view(nlev);
  auto cldt_dev = mam4::validation::create_column_view(nlev);
  auto cldc_dev = mam4::validation::create_column_view(nlev);
  auto cldvcu_dev = mam4::validation::create_column_view(nlev);
  auto cldvst_dev = mam4::validation::create_column_view(nlev);
  auto tracer_dev = mam4::validation::create_column_view(nlev);
  auto scavcoef_dev = mam4::validation::create_column_view(nlev);
  auto f_act_conv_dev = mam4::validation::create_column_view(nlev);
  auto qqcw_dev = mam4::validation::create_column_view(nlev);
  auto sol_factic_dev = mam4::validation::create_column_view(nlev);

  auto copy_to_dev = [](mam4::ColumnView dev, std::vector<Real> vec) {
    auto host = Kokkos::create_mirror_view(dev);
    for (int i = 0; i < nlev; ++i)
      host[i] = vec[i];
    Kokkos::deep_copy(dev, host);
  };
  copy_to_dev(pdel_dev, pdel);

  copy_to_dev(pdel_dev, pdel);
  copy_to_dev(cmfdqr_dev, cmfdqr);
  copy_to_dev(evapc_dev, evapc);
  copy_to_dev(dlf_dev, dlf);
  copy_to_dev(conicw_dev, conicw);
  copy_to_dev(precs_dev, precs);
  copy_to_dev(evaps_dev, evaps);
  copy_to_dev(cwat_dev, cwat);
  copy_to_dev(cldt_dev, cldt);
  copy_to_dev(cldc_dev, cldc);
  copy_to_dev(cldvcu_dev, cldvcu);
  copy_to_dev(cldvst_dev, cldvst);
  copy_to_dev(tracer_dev, tracer);
  copy_to_dev(scavcoef_dev, scavcoef);
  copy_to_dev(f_act_conv_dev, f_act_conv);
  copy_to_dev(qqcw_dev, qqcw);
  copy_to_dev(sol_factic_dev, sol_factic);

  auto fracis_dev = mam4::validation::create_column_view(nlev);
  auto scavt_dev = mam4::validation::create_column_view(nlev);
  auto iscavt_dev = mam4::validation::create_column_view(nlev);
  auto icscavt_dev = mam4::validation::create_column_view(nlev);
  auto isscavt_dev = mam4::validation::create_column_view(nlev);
  auto bcscavt_dev = mam4::validation::create_column_view(nlev);
  auto bsscavt_dev = mam4::validation::create_column_view(nlev);
  auto rcscavt_dev = mam4::validation::create_column_view(nlev);
  auto rsscavt_dev = mam4::validation::create_column_view(nlev);

  Kokkos::parallel_for(
      "wetdep::wetdepa_v2", nlev, KOKKOS_LAMBDA(const int kk) {
        Real precabs = 0;
        Real precabc = 0;
        Real scavabs = 0;
        Real scavabc = 0;
        Real precabs_base = 0;
        Real precabc_base = 0;
        Real precnums_base = 0;
        Real precnumc_base = 0;
        mam4::wetdep::wetdepa_v2(
            deltat, pdel_dev[kk], cmfdqr_dev[kk], evapc_dev[kk], dlf_dev[kk],
            conicw_dev[kk], precs_dev[kk], evaps_dev[kk], cwat_dev[kk],
            cldt_dev[kk], cldc_dev[kk], cldvcu_dev[kk], cldvst_dev[kk],
            sol_factb, sol_facti, sol_factic_dev[kk], mam_prevap_resusp_optcc,
            is_strat_cloudborne, scavcoef_dev[kk], f_act_conv_dev[kk],
            tracer_dev[kk], qqcw_dev[kk], precabs, precabc, scavabs, scavabc,
            precabs_base, precabc_base, precnums_base, precnumc_base,
            scavt_dev[kk], bcscavt_dev[kk], rcscavt_dev[kk], rsscavt_dev[kk]);
      });

  auto copy_to_host = [&](std::string name, mam4::ColumnView dev) {
    std::vector<Real> vec(nlev);
    auto host = Kokkos::create_mirror_view(dev);
    Kokkos::deep_copy(host, dev);
    for (int i = 0; i < nlev; ++i)
      vec[i] = host[i];
    output.set(name, vec);
  };
  // Create mirror views for output arrays
  copy_to_host("fracis", fracis_dev);
  copy_to_host("scavt", scavt_dev);
  copy_to_host("iscavt", iscavt_dev);
  copy_to_host("icscavt", icscavt_dev);
  copy_to_host("isscavt", isscavt_dev);
  copy_to_host("bcscavt", bcscavt_dev);
  copy_to_host("bsscavt", bsscavt_dev);
  copy_to_host("rcscavt", rcscavt_dev);
  copy_to_host("rsscavt", rsscavt_dev);
}

void test_wetdepa_v2(std::unique_ptr<Ensemble> &ensemble) {
  ensemble->process([&](const Input &input, Output &output) {
    test_wetdepa_v2_process(input, output);
  });
}
