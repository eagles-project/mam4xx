// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include "atmosphere_utils.hpp"
#include "testing.hpp"
#include <mam4xx/aero_modes.hpp>
#include <mam4xx/mam4.hpp>

#include <ekat/ekat_type_traits.hpp>
#include <ekat/logging/ekat_logger.hpp>
#include <ekat/mpi/ekat_comm.hpp>

#include <catch2/catch.hpp>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>

using namespace haero;
using namespace mam4;
TEST_CASE("test_constructor", "mam4_gasaerexch_process") {
  mam4::AeroConfig mam4_config;
  mam4::GasAerExchProcess::ProcessConfig process_config;
  mam4::GasAerExchProcess process(mam4_config, process_config);
  REQUIRE(process.name() == "MAM4 gas/aersol exchange");
  REQUIRE(process.aero_config() == mam4_config);
}

TEST_CASE("test_compute_tendencies", "mam4_gasaerexch_process") {
  const int num_mode = mam4::GasAerExch::num_mode;
  int nlev = 72;
  Real pblh = 1000;
  // these values correspond to a humid atmosphere with relative humidity
  // values approximately between 32% and 98%
  const Real Tv0 = 300;     // reference virtual temperature [K]
  const Real Gammav = 0.01; // virtual temperature lapse rate [K/m]
  const Real qv0 =
      0.015; // specific humidity at surface [kg h2o / kg moist air]
  const Real qv1 = 7.5e-4; // specific humidity lapse rate [1 / m]
  Atmosphere atm =
      mam4::init_atm_const_tv_lapse_rate(nlev, pblh, Tv0, Gammav, qv0, qv1);

  Surface sfc = mam4::testing::create_surface();
  mam4::Prognostics progs = mam4::testing::create_prognostics(nlev);
  mam4::Diagnostics diags = mam4::testing::create_diagnostics(nlev);
  for (int i = 0; i < num_mode; ++i)
    Kokkos::deep_copy(diags.wet_geometric_mean_diameter_i[i], 0.001);
  mam4::Tendencies tends = mam4::testing::create_tendencies(nlev);

  mam4::AeroConfig mam4_config;
  mam4::GasAerExchProcess::ProcessConfig process_config;
  mam4::GasAerExchProcess process(mam4_config, process_config);

  // Single-column dispatch.
  auto team_policy = ThreadTeamPolicy(1u, Kokkos::AUTO);
  Real t = 0.0, dt = 30.0;
  Kokkos::parallel_for(
      team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
        process.compute_tendencies(team, t, dt, atm, sfc, progs, diags, tends);
      });
}

TEST_CASE("test_multicol_compute_tendencies", "mam4_gasaerexch_process") {
  // Now we process multiple columns within a single dispatch (mc means
  // "multi-column").
  const int num_mode = mam4::GasAerExch::num_mode;
  int ncol = 8;
  DeviceType::view_1d<Atmosphere> mc_atm("mc_progs", ncol);
  DeviceType::view_1d<Surface> mc_sfc("mc_sfc", ncol);
  DeviceType::view_1d<mam4::Prognostics> mc_progs("mc_atm", ncol);
  DeviceType::view_1d<mam4::Diagnostics> mc_diags("mc_diags", ncol);
  DeviceType::view_1d<mam4::Tendencies> mc_tends("mc_tends", ncol);
  const int nlev = 72;
  const Real pblh = 1000;
  // these values correspond to a humid atmosphere with relative humidity
  // values approximately between 32% and 98%
  const Real Tv0 = 300;     // reference virtual temperature [K]
  const Real Gammav = 0.01; // virtual temperature lapse rate [K/m]
  const Real qv0 =
      0.015; // specific humidity at surface [kg h2o / kg moist air]
  const Real qv1 = 7.5e-4; // specific humidity lapse rate [1 / m]
  Atmosphere atmosphere =
      mam4::init_atm_const_tv_lapse_rate(nlev, pblh, Tv0, Gammav, qv0, qv1);
  Surface surface = mam4::testing::create_surface();
  mam4::Prognostics prognostics = mam4::testing::create_prognostics(nlev);
  mam4::Diagnostics diagnostics = mam4::testing::create_diagnostics(nlev);
  for (int i = 0; i < num_mode; ++i)
    Kokkos::deep_copy(diagnostics.wet_geometric_mean_diameter_i[i], 0.001);
  mam4::Tendencies tendencies = mam4::testing::create_tendencies(nlev);
  for (int icol = 0; icol < ncol; ++icol) {
    Kokkos::parallel_for(
        "Load multi-column views", 1, KOKKOS_LAMBDA(const int) {
          mc_atm(icol) = atmosphere;
          mc_sfc(icol) = surface;
          mc_progs(icol) = prognostics;
          mc_diags(icol) = diagnostics;
          mc_tends(icol) = tendencies;
        });
  }

  mam4::AeroConfig mam4_config;
  mam4::GasAerExchProcess::ProcessConfig process_config;
  mam4::GasAerExchProcess process(mam4_config, process_config);

  // Dispatch over all the above columns.
  auto team_policy = ThreadTeamPolicy(ncol, Kokkos::AUTO);
  Real t = 0.0, dt = 30.0;
  Kokkos::parallel_for(
      team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
        const int icol = team.league_rank();
        process.compute_tendencies(team, t, dt, mc_atm(icol), mc_sfc(icol),
                                   mc_progs(icol), mc_diags(icol),
                                   mc_tends(icol));
      });
}

TEST_CASE("mam_gasaerexch_1subarea_1gas_nonvolatile", "mam_gasaerexch") {

  // Since there does not seem to be a way to extract the internal epsilon
  // used by Approx from the class, the following is based on code found
  // in the catch_approx.cpp source file.
  const double epsilon = ekat::is_single_precision<Real>::value
                             // Single precision value since the check values
                             // were created using double
                             ? .0001
                             // Default value used in class Approx
                             : std::numeric_limits<float>::epsilon() * 100;

  ekat::Comm comm;
  ekat::logger::Logger<> logger("gasaerexch unit tests",
                                ekat::logger::LogLevel::debug, comm);

  const int num_mode = mam4::GasAerExch::num_mode;
  // Values as used in mam_refactor smoke test:

  const Real dts[3] = {1.0, 1.0e-20, 10.0};
  const Real qgas_netprod_otrproc = 5.00000000E-016;
  // clang-format off
  const Real uptkaer[3][10][num_mode] = {
      {{2.8825214836391422E-004, 1.6150479495487672E-004, 1.5516789173297778E-005, 9.7833674238607867E-005},
       {2.8913181177882068E-004, 1.7561477711278560E-004, 1.5516799808257626E-005, 9.7543918825297989E-005},
       {2.8995097552900125E-004, 1.8827585007002664E-004, 1.5516810433728653E-005, 9.7255542855016150E-005},
       {2.9076430264442552E-004, 1.9983549493319009E-004, 1.5516821049693655E-005, 9.6968537558659326E-005},
       {2.9157186993739688E-004, 2.1055830303835061E-004, 1.5516831656139272E-005, 9.6682894223796390E-005},
       {2.9237375274054260E-004, 2.2062176911155144E-004, 1.5516842253055303E-005, 9.6398604189172510E-005},
       {2.9317002495562544E-004, 2.3015115566096369E-004, 1.5516852840433991E-005, 9.6115658834033686E-005},
       {2.9396075908880262E-004, 2.3923836936107343E-004, 1.5516863418269646E-005, 9.5834049715385553E-005},
       {2.9474603063904981E-004, 2.4795295093232958E-004, 1.5516873986558298E-005, 9.5553766529866868E-005},
       {2.9552590525020128E-004, 2.5634885743190211E-004, 1.5516884545297504E-005, 9.5274802336758267E-005}},
      {{2.8825214836391422E-004, 1.6150479495487672E-004, 1.5516789173297778E-005, 9.7833674238607867E-005},
       {2.8890459238538460E-004, 1.7560634088654813E-004, 1.5516798632034082E-005, 9.7624646704106484E-005},
       {2.8950525373480109E-004, 1.8825095185645829E-004, 1.5516808088067353E-005, 9.7416122925779395E-005},
       {2.9010346874411020E-004, 1.9979314881714397E-004, 1.5516817541343080E-005, 9.7208102554154305E-005},
       {2.9069925485252976E-004, 2.1049749594694765E-004, 1.5516826991810652E-005, 9.7000585233351458E-005},
       {2.9129262933054939E-004, 2.2054153351763856E-004, 1.5516836439422636E-005, 9.6793570591100376E-005},
       {2.9188360929509333E-004, 2.3005060013274006E-004, 1.5516845884134081E-005, 9.6587058226920778E-005},
       {2.9247221171956977E-004, 2.3911669065468282E-004, 1.5516855325902192E-005, 9.6381047709155485E-005},
       {2.9305845780266997E-004, 2.4780943745878977E-004, 1.5516864764685890E-005, 9.6175536950290037E-005},
       {2.9364236001415585E-004, 2.5618288860993974E-004, 1.5516874200445654E-005, 9.5970527021146522E-005}},
      {{2.8825214836391422E-004, 1.6150479495487672E-004, 1.5516789173297778E-005, 9.7833674238607867E-005},
       {2.9116278226243395E-004, 1.7569034260608683E-004, 1.5516810365887290E-005, 9.6819315386397468E-005},
       {2.9389725469814988E-004, 1.8849827927330221E-004, 1.5516831453473513E-005, 9.5822476431747615E-005},
       {2.9656061262933964E-004, 2.0021260416943861E-004, 1.5516852435974999E-005, 9.4842834161189502E-005},
       {2.9915582261753838E-004, 2.1109790586903529E-004, 1.5516873313408740E-005, 9.3880065784608937E-005},
       {3.0168567232697940E-004, 2.2133105228309445E-004, 1.5516894085868569E-005, 9.2933850899594274E-005},
       {3.0415278387347158E-004, 2.3103642700450936E-004, 1.5516914753510029E-005, 9.2003872346993856E-005},
       {3.0655962636241189E-004, 2.4030496709980811E-004, 1.5516935316540132E-005, 9.1089816885070487E-005},
       {3.0890853150773560E-004, 2.4920522747950768E-004, 1.5516955775209287E-005, 9.0191374070883621E-005},
       {3.1120169225663243E-004, 2.5779019803550567E-004, 1.5516976129805324E-005, 8.9308241299504377E-005}}};

  const Real in_qaer_cur[3][10][num_mode] = {
      {{3.4543891238118817E-011, 2.6039003387624214E-012, 1.3841779609564249E-010, 0.0000000000000000},
       {3.4550145466431223E-011, 2.7085381034471938E-012, 1.3841797477113269E-010, 0.0000000000000000},
       {3.4554573528111080E-011, 2.8143998693114451E-012, 1.3841815241630738E-010, 0.0000000000000000},
       {3.4558982187451026E-011, 2.9196746222338241E-012, 1.3841832903692752E-010, 0.0000000000000000},
       {3.4563371464457567E-011, 3.0243599959047575E-012, 1.3841850463875586E-010, 0.0000000000000000},
       {3.4567741385312547E-011, 3.1284548767492370E-012, 1.3841867922755203E-010, 0.0000000000000000},
       {3.4572091978034984E-011, 3.2319590017286081E-012, 1.3841885280906893E-010, 0.0000000000000000},
       {3.4576423272432240E-011, 3.3348727122992813E-012, 1.3841902538904974E-010, 0.0000000000000000},
       {3.4580735316449740E-011, 3.4371967794683046E-012, 1.3841919697322532E-010, 0.0000000000000000},
       {3.4585028127638834E-011, 3.5389323450407532E-012, 1.3841936756731221E-010, 0.0000000000000000}},
      {{3.4543891238118817E-011, 2.6039003387624214E-012, 1.3841779609564249E-010, 0.0000000000000000},
       {3.4545523799680896E-011, 2.7068834670370362E-012, 1.3841779609564249E-010, 0.0000000000000000},
       {3.4545524974978544E-011, 2.8108004926354357E-012, 1.3841779609564252E-010, 0.0000000000000000},
       {3.4545526239890154E-011, 2.9140350335264169E-012, 1.3841779609564254E-010, 0.0000000000000000},
       {3.4545527590932610E-011, 3.0165991273645691E-012, 1.3841779609564252E-010, 0.0000000000000000},
       {3.4545529028638121E-011, 3.1185027921817353E-012, 1.3841779609564252E-010, 0.0000000000000000},
       {3.4545530553363484E-011, 3.2197546827103235E-012, 1.3841779609564249E-010, 0.0000000000000000},
       {3.4545532165360273E-011, 3.3203624823305110E-012, 1.3841779609564252E-010, 0.0000000000000000},
       {3.4545533881244394E-011, 3.4203331375492286E-012, 1.3841779609564249E-010, 0.0000000000000000},
       {3.4545535685489576E-011, 3.5196730751680504E-012, 1.3841779609564252E-010, 0.0000000000000000}},
      {{3.4543891238118817E-011, 2.6039003387624214E-012, 1.3841779609564249E-010, 0.0000000000000000},
       {3.4591644767609546E-011, 2.7233739106876547E-012, 1.3841957854019800E-010, 0.0000000000000000},
       {3.4635825790576510E-011, 2.8466406108906802E-012, 1.3842134545161804E-010, 0.0000000000000000},
       {3.4679814510300586E-011, 2.9701257437833903E-012, 1.3842309682673812E-010, 0.0000000000000000},
       {3.4723601695290640E-011, 3.0936906910013100E-012, 1.3842483267672178E-010, 0.0000000000000000},
       {3.4767178859352076E-011, 3.2172283346673429E-012, 1.3842655302389037E-010, 0.0000000000000000},
       {3.4810538176560682E-011, 3.3406528763779508E-012, 1.3842825789955799E-010, 0.0000000000000000},
       {3.4853672417570752E-011, 3.4638937045740747E-012, 1.3842994734248866E-010, 0.0000000000000000},
       {3.4896574913559294E-011, 3.5868914457980456E-012, 1.3843162139774970E-010, 0.0000000000000000},
       {3.4939239464266833E-011, 3.7095953649427108E-012, 1.3843328011583073E-010, 0.0000000000000000}}};

  const Real out_qaer_cur[3][10][num_mode] = {
      {{3.4549423266828319E-011, 2.6069998791040425E-012, 1.3841809388812617E-010, 1.8775877221248125E-015},
       {3.4555662376258777E-011, 2.7118890003400572E-012, 1.3841827084642385E-010, 1.8612306998486968E-015},
       {3.4560074156105117E-011, 2.8179716297085476E-012, 1.3841844678400762E-010, 1.8450241825440182E-015},
       {3.4564466423235679E-011, 2.9234438090369814E-012, 1.3841862170664142E-010, 1.8289670321569599E-015},
       {3.4568839202426779E-011, 3.0283085168251816E-012, 1.3841879562008280E-010, 1.8130580698129665E-015},
       {3.4573192524433275E-011, 3.1325682417322939E-012, 1.3841896853008018E-010, 1.7972960895205187E-015},
       {3.4577526421668389E-011, 3.2362252751403981E-012, 1.3841914044237027E-010, 1.7816798640992575E-015},
       {3.4581840928172393E-011, 3.3392818420384053E-012, 1.3841931136267571E-010, 1.7662081536054488E-015},
       {3.4586136096057272E-011, 3.4417401459663029E-012, 1.3841948129670346E-010, 1.7508796728450361E-015},
       {3.4590411946750202E-011, 3.5436024462173347E-012, 1.3841965025014331E-010, 1.7356931914926501E-015}},
      {{3.4543891238118817E-011, 2.6039003387624214E-012, 1.3841779609564249E-010, 0.0000000000000000},
       {3.4545523799680896E-011, 2.7068834670370362E-012, 1.3841779609564249E-010, 0.0000000000000000},
       {3.4545524974978544E-011, 2.8108004926354357E-012, 1.3841779609564252E-010, 0.0000000000000000},
       {3.4545526239890154E-011, 2.9140350335264169E-012, 1.3841779609564254E-010, 0.0000000000000000},
       {3.4545527590932610E-011, 3.0165991273645691E-012, 1.3841779609564252E-010, 0.0000000000000000},
       {3.4545529028638121E-011, 3.1185027921817353E-012, 1.3841779609564252E-010, 0.0000000000000000},
       {3.4545530553363484E-011, 3.2197546827103235E-012, 1.3841779609564249E-010, 0.0000000000000000},
       {3.4545532165360273E-011, 3.3203624823305110E-012, 1.3841779609564252E-010, 0.0000000000000000},
       {3.4545533881244394E-011, 3.4203331375492286E-012, 1.3841779609564249E-010, 0.0000000000000000},
       {3.4545535685489576E-011, 3.5196730751680504E-012, 1.3841779609564252E-010, 0.0000000000000000}},
      {{3.4599078071202227E-011, 2.6348209692310381E-012, 1.3842076683656830E-010, 1.8730582515300600E-014},
       {3.4646902993251958E-011, 2.7567172402144687E-012, 1.3842252339256473E-010, 1.8374819523957843E-014},
       {3.4691112457969225E-011, 2.8821000835590644E-012, 1.3842426441015149E-010, 1.8025705577513406E-014},
       {3.4735107589949117E-011, 3.0074549473172955E-012, 1.3842598991004419E-010, 1.7683239648285265E-014},
       {3.4778880429824748E-011, 3.1326978712632150E-012, 1.3842769992200281E-010, 1.7347385015423203E-014},
       {3.4822423652455173E-011, 3.2577585597070562E-012, 1.3842939448333645E-010, 1.7018081520481834E-014},
       {3.4865730502560982E-011, 3.3825773251197445E-012, 1.3843107363777577E-010, 1.6695253126394671E-014},
       {3.4908794745193754E-011, 3.5071028162339715E-012, 1.3843273743459038E-010, 1.6378812790964114E-014},
       {3.4951610640714727E-011, 3.6312903195951770E-012, 1.3843438592788476E-010, 1.6068665474893838E-014},
       {3.4994172858562415E-011, 3.7551005499933817E-012, 1.3843601917602618E-010, 1.5764711295672193E-014}}};

  const Real out_qgas_avg[3][10] = {
      {1.9191631470368576E-011, 1.9080951697012725E-011, 1.8970890869739010E-011, 1.8861449178322882E-011,
       1.8752626288706048E-011, 1.8644421479455502E-011, 1.8536833700859000E-011, 1.8429861632514369E-011,
       1.8323503731934381E-011, 1.8217758273907820E-011},
      {1.9196785428799816E-011, 1.9092369738963115E-011, 1.8988651538067063E-011, 1.8885615732264482E-011,
       1.8783250287383871E-011, 1.8681545184861183E-011, 1.8580491769607234E-011, 1.8480082357990262E-011,
       1.8380309986887433E-011, 1.8281168245023424E-011},
      {1.9145332791670708E-011, 1.8978464628287798E-011, 1.8811563057810087E-011, 1.8644781975021802E-011,
       1.8478241222395088E-011, 1.8312037385460390E-011, 1.8146250478923765E-011, 1.7980948201519970E-011,
       1.7816188787926866E-011, 1.7652023000658613E-011}};

  const Real in_qgas_cur[3][10] = {
      {1.9196785428799816E-011, 1.9086214760312434E-011, 1.8976247287593632E-011, 1.8866887254711152E-011,
       1.8758137002205340E-011, 1.8649997611709737E-011, 1.8542469312491037E-011, 1.8435551727542304E-011,
       1.8329244032180201E-011, 1.8223545061331766E-011},
      {1.9196785428799816E-011, 1.9092369738963115E-011, 1.8988651538067063E-011, 1.8885615732264482E-011,
       1.8783250287383871E-011, 1.8681545184861183E-011, 1.8580491769607234E-011, 1.8480082357990262E-011,
       1.8380309986887433E-011, 1.8281168245023424E-011},
      {1.9196785428799816E-011, 1.9030975882828376E-011, 1.8864961248238360E-011, 1.8698936020501514E-011,
       1.8533048038309896E-011, 1.8367412883413813E-011, 1.8202124148826970E-011, 1.8037259636690117E-011,
       1.7872885344216572E-011, 1.7709058156283342E-011}};

  const Real out_qgas_cur[3][10] = {
      {1.9186478479542893E-011, 1.9075689647601013E-011, 1.8965535507319710E-011, 1.8856012195377293E-011,
       1.8747116703918971E-011, 1.8638846508988273E-011, 1.8531199282280415E-011, 1.8424172760283460E-011,
       1.8317764682923683E-011, 1.8211972765021210E-011},
      {1.9196785428799816E-011, 1.9092369738963115E-011, 1.8988651538067063E-011, 1.8885615732264482E-011,
       1.8783250287383871E-011, 1.8681545184861183E-011, 1.8580491769607234E-011, 1.8480082357990262E-011,
       1.8380309986887433E-011, 1.8281168245023424E-011},
      {1.9093976641806687E-011, 1.8926054655768472E-011, 1.8758270444066314E-011, 1.8590737414364731E-011,
       1.8423547493217444E-011, 1.8256778324304468E-011, 1.8090496382740719E-011, 1.7924759292514538E-011,
       1.7759617547654076E-011, 1.7595115805445990E-011}};
  // clang-format on

  for (int p = 0; p < 3; ++p) {
    const Real dt = dts[p];
    for (int n = 0; n < 10; ++n) {
      Real qaer_cur[num_mode];
      Real qgas_cur = in_qgas_cur[p][n];
      Real qgas_avg = 0;
      for (int j = 0; j < num_mode; j++)
        qaer_cur[j] = in_qaer_cur[p][n][j];
      gasaerexch::mam_gasaerexch_1subarea_1gas_nonvolatile(
          dt, qgas_netprod_otrproc, uptkaer[p][n], qgas_cur, qgas_avg,
          qaer_cur);

      if (!(qgas_cur == Approx(out_qgas_cur[p][n]).epsilon(epsilon)))
        std::cout << "qgas_cur != Approx(test_qgas_cur)): "
                  << std::setprecision(14) << qgas_cur
                  << " != " << out_qgas_cur[p][n] << std::endl;
      REQUIRE(qgas_cur == Approx(out_qgas_cur[p][n]).epsilon(epsilon));

      if (!(qgas_avg == Approx(out_qgas_avg[p][n]).epsilon(epsilon)))
        std::cout << "qgas_avg != Approx(test_qgas_avg)): "
                  << std::setprecision(14) << qgas_avg
                  << " != " << out_qgas_avg[p][n] << std::endl;
      REQUIRE(qgas_avg == Approx(out_qgas_avg[p][n]).epsilon(epsilon));

      for (int i = 0; i < num_mode; ++i) {
        if (!(qaer_cur[i] == Approx(out_qaer_cur[p][n][i]).epsilon(epsilon)))
          std::cout << "qaer_cur[i] != Approx(test_qaer_cur[i])): "
                    << std::setprecision(14) << qaer_cur[i]
                    << " != " << out_qaer_cur[p][n][i] << " (n,i):(" << n << ","
                    << i << ")" << std::endl;
        REQUIRE(qaer_cur[i] == Approx(out_qaer_cur[p][n][i]).epsilon(epsilon));
      }
    }
  }
}
