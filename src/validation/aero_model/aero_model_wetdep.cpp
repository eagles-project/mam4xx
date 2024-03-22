// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/mam4.hpp>

#include <mam4xx/aero_config.hpp>
#include <skywalker.hpp>
#include <validation.hpp>
#include <mam4xx/mam4.hpp>
using namespace skywalker;
using namespace mam4;
using namespace haero;
using namespace haero::testing;


void aero_model_wetdep(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {

mam4::Prognostics progs = validation::create_prognostics(nlev);
int nlev = mam4::nlev;
Real pblh = 1000;
Atmosphere atm = validation::create_atmosphere(nlev, pblh);
// FIXME get this yaml 
const Real dt=0;

// inputs 
auto cldn_prev_step =  create_column_view(nlev);
auto rprdsh =  create_column_view(nlev);
auto rprddp =  create_column_view(nlev);
auto evapcdp =  create_column_view(nlev);
auto evapcsh =  create_column_view(nlev);
auto dp_frac =  create_column_view(nlev);
auto sh_frac =  create_column_view(nlev);
auto dp_ccf =  create_column_view(nlev);
auto sh_ccf =  create_column_view(nlev);
auto icwmrdp =  create_column_view(nlev);
auto icwmrsh =  create_column_view(nlev);
auto evapr =  create_column_view(nlev);
auto cldst =  create_column_view(nlev);

// outputs 
auto dlf =  create_column_view(nlev);
auto aerdepwetis =  create_column_view(nlev);
auto aerdepwetcw =  create_column_view(nlev);

const int work_len = wetdep::get_aero_model_wetdep_work_len();
wetdep::View1D work("work", work_len);

Kokkos::View<Real * [aero_model::maxd_aspectype + 2][aero_model::pcnst]>
                       qqcw_sav("qqcw_sav", nlev);                                           

auto team_policy = ThreadTeamPolicy(1u, Kokkos::AUTO);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {

    auto   progs_in =   progs;
    wetdep::aero_model_wetdep(team, 
                      atm,
                      progs_in,
                      dt,
                       // inputs
                      cldn_prev_step, 
                      rprdsh,
                      rprddp,
                      evapcdp,
                      evapcsh,
                      dp_frac,
                      sh_frac,
                      dp_ccf,
                      sh_ccf,
                      icwmrdp,
                      icwmrsh,
                      evapr,
                      cldst,
                       // output 
                      dlf,
                      aerdepwetis,
                      aerdepwetcw,
                       // FIXME 
                      qqcw_sav,
                      work);

 });


  });
}    