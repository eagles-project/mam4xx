// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/mam4.hpp>

#include <mam4xx/aero_config.hpp>
#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;
using namespace haero;
using namespace gasaerexch;

void mam_gasaerexch_1subarea(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    // Extract scalar values from input arrays
    const int jtsubstep = input.get_array("jtsubstep")[0];
    const Real dtsubstep = input.get_array("dtsubstep")[0];
    const Real temp = input.get_array("temp")[0];
    const Real pmid = input.get_array("pmid")[0];
    const Real aircon = input.get_array("aircon")[0];
    const int n_mode = input.get_array("n_mode")[0];

    // Extract array values from input arrays
    auto qgas_cur_vec = input.get_array("qgas_cur");
    auto qgas_avg_vec = input.get_array("qgas_avg");
    auto qgas_netprod_otrproc_vec = input.get_array("qgas_netprod_otrproc");
    auto qaer_cur_vec = input.get_array("qaer_cur");
    auto qnum_cur_vec = input.get_array("qnum_cur");
    auto qwtr_cur_vec = input.get_array("qwtr_cur");
    auto dgn_a_vec = input.get_array("dgn_a");
    auto dgn_awet_vec = input.get_array("dgn_awet");
    auto wetdens_vec = input.get_array("wetdens");
    auto uptkaer_vec = input.get_array("uptkaer");

    Real qgas_cur[gasaerexch::max_gas];
    Real qgas_avg[gasaerexch::max_gas];
    Real qgas_netprod_otrproc[gasaerexch::max_gas];
    Real qaer_cur[gasaerexch::max_aer][mam4::gasaerexch::max_mode];
    Real qnum_cur[gasaerexch::max_mode];
    Real qwtr_cur[gasaerexch::max_mode];
    Real dgn_a[gasaerexch::max_mode];
    Real dgn_awet[gasaerexch::max_mode];
    Real wetdens[gasaerexch::max_mode];
    Real uptkaer[gasaerexch::max_gas][mam4::gasaerexch::max_mode];
    Real uptkrate_h2so4=0;

    // Copy input vectors to local arrays
    for (int i = 0; i < gasaerexch::max_gas; ++i) {
      qgas_cur[i] = qgas_cur_vec[i];
      qgas_avg[i] = qgas_avg_vec[i];
      qgas_netprod_otrproc[i] = qgas_netprod_otrproc_vec[i];
    }

    for (int i = 0; i < gasaerexch::max_mode; ++i) {
      qnum_cur[i] = qnum_cur_vec[i];
      qwtr_cur[i] = qwtr_cur_vec[i];
      dgn_a[i] = dgn_a_vec[i];
      dgn_awet[i] = dgn_awet_vec[i];
      wetdens[i] = wetdens_vec[i];
    }

    
    for (int j = 0; j < mam4::gasaerexch::max_mode; ++j) {
      for (int i = 0; i < gasaerexch::max_aer; ++i) {  
        qaer_cur[i][j] = qaer_cur_vec[j * mam4::gasaerexch::max_aer + i];
      }
    }

    for (int j = 0; j < mam4::gasaerexch::max_mode; ++j) {
      for (int i = 0; i < gasaerexch::max_gas; ++i) {
        uptkaer[i][j] = uptkaer_vec[j * mam4::gasaerexch::max_gas + i];
      }
    }

    // Call the function
    mam_gasaerexch_1subarea(jtsubstep, dtsubstep, temp, pmid, aircon, n_mode,
                            qgas_cur, qgas_avg, qgas_netprod_otrproc, qaer_cur,
                            qnum_cur, qwtr_cur, dgn_a, dgn_awet, wetdens, uptkaer,
                            uptkrate_h2so4);

    // Prepare output arrays
    std::vector<Real> qgas_cur_out(gasaerexch::max_gas);
    std::vector<Real> qgas_avg_out(gasaerexch::max_gas);
    std::vector<Real> qaer_cur_out(gasaerexch::max_aer * mam4::gasaerexch::max_mode);
    std::vector<Real> qnum_cur_out(gasaerexch::max_mode);
    std::vector<Real> qwtr_cur_out(gasaerexch::max_mode);
    std::vector<Real> dgn_a_out(gasaerexch::max_mode);
    std::vector<Real> dgn_awet_out(gasaerexch::max_mode);
    std::vector<Real> wetdens_out(gasaerexch::max_mode);
    std::vector<Real> uptkaer_out(gasaerexch::max_gas * mam4::gasaerexch::max_mode);

    for (int i = 0; i < gasaerexch::max_gas; ++i) {
      qgas_cur_out[i] = qgas_cur[i];
      qgas_avg_out[i] = qgas_avg[i];
    }

    for (int i = 0; i < gasaerexch::max_mode; ++i) {
      qnum_cur_out[i] = qnum_cur[i];
      qwtr_cur_out[i] = qwtr_cur[i];
      dgn_a_out[i] = dgn_a[i];
      dgn_awet_out[i] = dgn_awet[i];
      wetdens_out[i] = wetdens[i];
    }

    for (int j = 0; j < mam4::gasaerexch::max_mode; ++j) {
      for (int i = 0; i < gasaerexch::max_aer; ++i) {
        qaer_cur_out[j * mam4::gasaerexch::max_aer + i] = qaer_cur[i][j];
      }
    }

    for (int j = 0; j < mam4::gasaerexch::max_mode; ++j) {
      for (int i = 0; i < gasaerexch::max_gas; ++i) {
        uptkaer_out[j * mam4::gasaerexch::max_gas + i] = uptkaer[i][j];
      }
    }

    // Set output values
    output.set("qgas_cur", qgas_cur_out);
    output.set("qgas_avg", qgas_avg_out);
    output.set("qaer_cur", qaer_cur_out);
    output.set("qnum_cur", qnum_cur_out);
    output.set("qwtr_cur", qwtr_cur_out);
    output.set("dgn_a", dgn_a_out);
    output.set("dgn_awet", dgn_awet_out);
    output.set("wetdens", wetdens_out);
    output.set("uptkaer", uptkaer_out);
    output.set("uptkrate_h2so4", uptkrate_h2so4);
  });
}