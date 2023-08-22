#include "atmosphere_utils.hpp"
#include "testing.hpp"

#include <catch2/catch.hpp>
#include <ekat/ekat_pack_kokkos.hpp>
#include <ekat/logging/ekat_logger.hpp>
#include <ekat/mpi/ekat_comm.hpp>
#include <haero/floating_point.hpp>
#include <mam4xx/aero_config.hpp>
#include <mam4xx/gas_chem.hpp>
#include <mam4xx/mam4.hpp>

using namespace skywalker;
using namespace mam4;
using namespace gas_chemistry;

// #include "mam4xx/aero_modes.hpp"
// #include "mam4xx/conversions.hpp"
// #include <haero/constants.hpp>
// #include <haero/haero.hpp>

// NOTE: other than the nonnegative-ish requirements, this test is basically
// vacuous, since it mostly does the same thing as the function, but I suppose
// it'll let us know if the function changes while not having to worry about
// new/globally-defined constants
TEST_CASE("test_solver_4_failure", "mam4_gaschem_unit_tests") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger("gaschem unit tests",
                                ekat::logger::LogLevel::debug, comm);
  const Real zero = 0;
  // const auto lin_jac = input.get_array("lin_jac");
  // const auto lrxt = input.get_array("lrxt");
  // const auto lhet = input.get_array("lhet");
  // const auto iter_invariant = input.get_array("iter_invariant");
  // auto lsol = input.get_array("lsol");
  // auto solution = input.get_array("solution");
  const Real dti = 0.2777777778e-3;

  std::vector<Real> prod(clscnt4, zero);
  std::vector<Real> loss(clscnt4, zero);
  std::vector<Real> max_delta(clscnt4, zero);

  Real epsilon[clscnt4] = {};
  imp_slv_inti(epsilon);

  // const int rxntot = 7;     // number of total reactions
  // const int gas_pcnst = 31; // number of gas phase species
  // const int nzcnt = 32;     // number of non-zero matrix entries
  // const int clscnt4 = 30;   // number of species in implicit class
  // const int extcnt = 9;     // number of species with external forcing
  // const int nfs = 8;        // number of fixed species
  // const int permute_4[gas_pcnst] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
  //                                   10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
  //                                   20, 21, 22, 23, 24, 25, 26, 27, 28, 29};
  // const int clsmap_4[gas_pcnst] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
  //                                  11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
  //                                  21, 22, 23, 24, 25, 26, 27, 28, 29, 30};

  dt:    0.00
  dti: [0.2777777778E-003]
  lin_jac: [-.3876940583E-004,-.4444443653E-005,0.1771299913E-005,-.3712088560E-004,0.2646782869E-004,-.3116930588E-004,0.0000000000E+000,0.0000000000E+000,0.0000000000E+000,0.0000000000E+000,0.0000000000E+000,0.0000000000E+000,0.0000000000E+000,0.0000000000E+000,0.0000000000E+000,0.0000000000E+000,0.0000000000E+000,0.0000000000E+000,0.0000000000E+000,0.0000000000E+000,0.0000000000E+000,0.0000000000E+000,0.0000000000E+000,0.0000000000E+000,0.0000000000E+000,0.0000000000E+000,0.0000000000E+000,0.0000000000E+000,0.0000000000E+000,0.0000000000E+000,0.0000000000E+000,0.0000000000E+000]
  lrxt: [0.0000000000E+000,0.5166088810E-014,0.3419820148E-005,0.1771299913E-005,0.8695113196E-005,0.9402954384E-005,0.1307123830E-004]
  lhet: [0.0000000000E+000,0.3534958569E-004,0.4444443653E-005,0.3534958569E-004,0.0000000000E+000,0.0000000000E+000,0.0000000000E+000,0.0000000000E+000,0.0000000000E+000,0.0000000000E+000,0.0000000000E+000,0.0000000000E+000,0.0000000000E+000,0.0000000000E+000,0.0000000000E+000,0.0000000000E+000,0.0000000000E+000,0.0000000000E+000,0.0000000000E+000,0.0000000000E+000,0.0000000000E+000,0.0000000000E+000,0.0000000000E+000,0.0000000000E+000,0.0000000000E+000,0.0000000000E+000,0.0000000000E+000,0.0000000000E+000,0.0000000000E+000,0.0000000000E+000,0.0000000000E+000]
  iter_invariant: [0.3540757898E-013,0.1322925052E-017,0.7600857521E-015,0.7078370470E-016,0.1032767998E-013,0.3162618100E-013,0.6921437694E-013,0.6258597416E-012,0.9416920102E-014,0.1273836795E-015,0.2485200214E-014,0.1080994459E-018,0.1800909531E+007,0.9778844460E-015,0.3052719355E-013,0.3441745081E-018,0.1909915045E-022,0.4470095034E+007,0.4058339613E-015,0.1927095488E-013,0.3715469722E-014,0.1162980074E-014,0.7379871842E-014,0.4628399603E-013,0.1421930842E-019,0.2233050316E+003,0.1493511427E-014,0.3416929371E-015,0.7539655251E-025,0.1383049154E+005]
  lsol: [0.3535379752E-007,0.1088693646E-009,0.4762530189E-014,0.2736308593E-011,0.2548213369E-012,0.3549685501E-010,0.1138542516E-009,0.2491717570E-009,0.2253095070E-008,0.3390091237E-010,0.4585812463E-012,0.8946720770E-011,0.3891580054E-015,0.6483274310E+010,0.3520384006E-011,0.1098978968E-009,0.1239028229E-014,0.6875694162E-019,0.1609234212E+011,0.1461002261E-011,0.6937543758E-010,0.1337569100E-010,0.4186728267E-011,0.2656753863E-010,0.1666223857E-009,0.5118951031E-016,0.8038981136E+006,0.5376636899E-011,0.1230094141E-011,0.2714275891E-021,0.4978972675E+008]
  solution: [0.1088693646E-009,0.4762530189E-014,0.2736308593E-011,0.2548213369E-012,0.3549685501E-010,0.1138542516E-009,0.2491717570E-009,0.2253095070E-008,0.3390091237E-010,0.4585812463E-012,0.8946720770E-011,0.3891580054E-015,0.6483274310E+010,0.3520384006E-011,0.1098978968E-009,0.1239028229E-014,0.6875694162E-019,0.1609234212E+011,0.1461002261E-011,0.6937543758E-010,0.1337569100E-010,0.4186728267E-011,0.2656753863E-010,0.1666223857E-009,0.5118951031E-016,0.8038981136E+006,0.5376636899E-011,0.1230094141E-011,0.2714275891E-021,0.4978972675E+008]


  mat[0] = lmat[0] - dti;
  mat[1] = lmat[1] - dti;
  mat[2] = lmat[2];
  mat[3] = lmat[3] - dti;
  mat[4] = lmat[4];
  mat[5] = lmat[5] - dti;
  mat[6] = lmat[6] - dti;
  mat[7] = lmat[7] - dti;
  mat[8] = lmat[8] - dti;
  mat[9] = lmat[9] - dti;
  mat[10] = lmat[10] - dti;
  mat[11] = lmat[11] - dti;
  mat[12] = lmat[12] - dti;
  mat[13] = lmat[13] - dti;
  mat[14] = lmat[14] - dti;
  mat[15] = lmat[15] - dti;
  mat[16] = lmat[16] - dti;
  mat[17] = lmat[17] - dti;
  mat[18] = lmat[18] - dti;
  mat[19] = lmat[19] - dti;
  mat[20] = lmat[20] - dti;
  mat[21] = lmat[21] - dti;
  mat[22] = lmat[22] - dti;
  mat[23] = lmat[23] - dti;
  mat[24] = lmat[24] - dti;
  mat[25] = lmat[25] - dti;
  mat[26] = lmat[26] - dti;
  mat[27] = lmat[27] - dti;
  mat[28] = lmat[28] - dti;
  mat[29] = lmat[29] - dti;
  mat[30] = lmat[30] - dti;
  mat[31] = lmat[31] - dti;

  const bool factor[itermax] = {true};
  bool converged[clscnt4] = {true};
  bool convergence = false;

  // void newton_raphson_iter(const Real dti, const Real lin_jac[nzcnt],
  //                          const Real lrxt[rxntot],
  //                          const Real lhet[gas_pcnst],         // in
  //                          const Real iter_invariant[clscnt4], // in
  //                          const bool factor[itermax],
  //                          const int permute_4[gas_pcnst],
  //                          const int clsmap_4[gas_pcnst], Real lsol[gas_pcnst],
  //                          Real solution[clscnt4],                     // inout
  //                          bool converged[clscnt4], bool &convergence, // out
  //                          Real prod[clscnt4], Real loss[clscnt4],
  //                          Real max_delta[clscnt4],
  //                          // work array
  //                          Real epsilon[clscnt4])

  newton_raphson_iter(dti, lin_jac.data(), lrxt.data(),
                      lhet.data(),           // & ! in
                      iter_invariant.data(), //              & ! in
                      factor, permute_4, clsmap_4, lsol.data(),
                      solution.data(),        //              & ! inout
                      converged, convergence, //         & ! out
                      prod.data(), loss.data(), max_delta.data(),
                      // work arrays
                      epsilon);

  std::vector<Real> converged_out;
  for (int i = 0; i < clscnt4; ++i) {
    if (converged[i]) {
      converged_out.push_back(1);
    } else {
      converged_out.push_back(0);
    }
  }

  std::vector<Real> convergence_out;

  if (convergence) {
    convergence_out.push_back(1);
  } else {
    convergence_out.push_back(0);
  }

}
