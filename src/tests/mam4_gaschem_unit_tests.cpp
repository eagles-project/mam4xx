#include "testing.hpp"

#include <catch2/catch.hpp>
#include <ekat/logging/ekat_logger.hpp>
#include <ekat/mpi/ekat_comm.hpp>
#include <mam4xx/gas_chem.hpp>
#include <mam4xx/mam4.hpp>

using namespace mam4;
using namespace gas_chemistry;

// all of these tests use the data from
// validation/gas_chem/newton_raphson_iter.cpp
// the first test uses the standard error tolerance of 1.0e-3
// and all entries converge
// the second test uses a ridiculously small error tolerance of 1.0e-35
// (to be close to the smallest single-precision number)
// and all entries fail to converge
// the third test uses a an error tolerance of 1.0e-17
// and results in 25 failures on the machine this test was written on
// (macbook pro with M1 Max, macos 13.5.2)

TEST_CASE("test_solver_4_full_converge", "mam4_gaschem_unit_tests") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger("gaschem unit tests",
                                ekat::logger::LogLevel::debug, comm);
  logger.debug("===============================================");
  logger.debug("***gas chem unit tests Case 1: full converge***");
  logger.debug("===============================================");
  const Real zero = 0;
  const Real dti = 0.2777777778e-3;
  const Real eps_val = 1.0e-3;

  logger.debug("case 1 error tolerance = {}", eps_val);

  std::vector<Real> prod(clscnt4, zero);
  std::vector<Real> loss(clscnt4, zero);
  std::vector<Real> max_delta(clscnt4, zero);

  Real epsilon[clscnt4];
  bool factor[itermax];
  bool converged[clscnt4];
  bool convergence = false;

  for (int i = 0; i < clscnt4; ++i) {
    epsilon[i] = eps_val;
    converged[i] = true;
  }
  for (int i = 0; i < itermax; ++i) {
    factor[i] = true;
  }

  const int gas_pcnst = 31; // number of gas phase species
  const int clscnt4 = 30;   // number of species in implicit class
  const int permute_4[gas_pcnst] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
                                    10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                    20, 21, 22, 23, 24, 25, 26, 27, 28, 29};

  const int clsmap_4[gas_pcnst] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                   11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                   21, 22, 23, 24, 25, 26, 27, 28, 29, 30};

  std::vector<Real> lin_jac = {
      -0.3876940583e-4, -0.4444443653e-5, 0.1771299913e-5, -0.3712088560e-4,
      0.2646782869e-4,  -0.3116930588e-4, 0.0000000000,    0.0000000000,
      0.0000000000,     0.0000000000,     0.0000000000,    0.0000000000,
      0.0000000000,     0.0000000000,     0.0000000000,    0.0000000000,
      0.0000000000,     0.0000000000,     0.0000000000,    0.0000000000,
      0.0000000000,     0.0000000000,     0.0000000000,    0.0000000000,
      0.0000000000,     0.0000000000,     0.0000000000,    0.0000000000,
      0.0000000000,     0.0000000000,     0.0000000000,    0.0000000000};

  std::vector<Real> lrxt = {0.0000000000,    0.5166088810e-14, 0.3419820148e-5,
                            0.1771299913e-5, 0.8695113196e-5,  0.9402954384e-5,
                            0.1307123830e-4};

  std::vector<Real> lhet = {
      0.0000000000, 0.3534958569e-4, 0.4444443653e-5, 0.3534958569e-4,
      0.0000000000, 0.0000000000,    0.0000000000,    0.0000000000,
      0.0000000000, 0.0000000000,    0.0000000000,    0.0000000000,
      0.0000000000, 0.0000000000,    0.0000000000,    0.0000000000,
      0.0000000000, 0.0000000000,    0.0000000000,    0.0000000000,
      0.0000000000, 0.0000000000,    0.0000000000,    0.0000000000,
      0.0000000000, 0.0000000000,    0.0000000000,    0.0000000000,
      0.0000000000, 0.0000000000,    0.0000000000};

  std::vector<Real> iter_invariant = {
      0.3540757898e+13, 0.1322925052e-17, 0.7600857521e+15, 0.7078370470e-16,
      0.1032767998e+13, 0.3162618100e-13, 0.6921437694e+13, 0.6258597416e-12,
      0.9416920102e-14, 0.1273836795e-15, 0.2485200214e-14, 0.1080994459e-18,
      0.1800909531e7,   0.9778844460e-15, 0.3052719355e-13, 0.3441745081e-18,
      0.1909915045e-22, 0.4470095034e7,   0.4058339613e-15, 0.1927095488e-13,
      0.3715469722e-14, 0.1162980074e-14, 0.7379871842e-14, 0.4628399603e-13,
      0.1421930842e-19, 0.2233050316e3,   0.1493511427e-14, 0.3416929371e-15,
      0.7539655251e-25, 0.1383049154e5};

  std::vector<Real> lsol = {
      0.3535379752e-7,  0.1088693646e-9,  0.4762530189e-14, 0.2736308593e-11,
      0.2548213369e-12, 0.3549685501e-10, 0.1138542516e-9,  0.2491717570e-9,
      0.2253095070e-8,  0.3390091237e-10, 0.4585812463e-12, 0.8946720770e-11,
      0.3891580054e-15, 0.6483274310e10,  0.3520384006e-11, 0.1098978968e-9,
      0.1239028229e-14, 0.6875694162e-19, 0.1609234212e11,  0.1461002261e-11,
      0.6937543758e-10, 0.1337569100e-10, 0.4186728267e-11, 0.2656753863e-10,
      0.1666223857e-9,  0.5118951031e-16, 0.8038981136e6,   0.5376636899e-11,
      0.1230094141e-11, 0.2714275891e-21, 0.4978972675e8};
  std::vector<Real> solution = {
      0.1088693646e-9,  0.4762530189e-14, 0.2736308593e-11, 0.2548213369e-12,
      0.3549685501e-10, 0.1138542516e-9,  0.2491717570e-9,  0.2253095070e-8,
      0.3390091237e-10, 0.4585812463e-12, 0.8946720770e-11, 0.3891580054e-15,
      0.6483274310e10,  0.3520384006e-11, 0.1098978968e-9,  0.1239028229e-14,
      0.6875694162e-19, 0.1609234212e11,  0.1461002261e-11, 0.6937543758e-10,
      0.1337569100e-10, 0.4186728267e-11, 0.2656753863e-10, 0.1666223857e-9,
      0.5118951031e-16, 0.8038981136e6,   0.5376636899e-11, 0.1230094141e-11,
      0.2714275891e-21, 0.4978972675e8};

  newton_raphson_iter(dti, lin_jac.data(), lrxt.data(), lhet.data(),
                      iter_invariant.data(), factor, permute_4, clsmap_4,
                      lsol.data(), solution.data(), converged, convergence,
                      prod.data(), loss.data(), max_delta.data(), epsilon);

  int conv_count = 0;
  for (int i = 0; i < clscnt4; ++i) {
    if (!converged[i]) {
      conv_count += 1;
      logger.debug("Did not converge for i = {} and max_delta[{}] = {}", i, i,
                   max_delta[i]);
    }
  }
  logger.debug("{} entries failed to converge", conv_count);
  REQUIRE(conv_count == 0);
}

TEST_CASE("test_solver_4_full_failure", "mam4_gaschem_unit_tests") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger("gaschem unit tests",
                                ekat::logger::LogLevel::debug, comm);
  logger.debug("===============================================");
  logger.debug("\n **gas chem unit tests Case 2: full failure**");
  logger.debug("===============================================");
  const Real zero = 0;
  const Real dti = 0.2777777778e-3;
  const Real eps_val = 1.0e-35;

  logger.debug("case 2 error tolerance = {}", eps_val);

  std::vector<Real> prod(clscnt4, zero);
  std::vector<Real> loss(clscnt4, zero);
  std::vector<Real> max_delta(clscnt4, zero);

  Real epsilon[clscnt4];
  bool factor[itermax];
  bool converged[clscnt4];
  bool convergence = false;

  for (int i = 0; i < clscnt4; ++i) {
    epsilon[i] = eps_val;
    converged[i] = true;
  }
  for (int i = 0; i < itermax; ++i) {
    factor[i] = true;
  }

  const int gas_pcnst = 31; // number of gas phase species
  const int clscnt4 = 30;   // number of species in implicit class
  const int permute_4[gas_pcnst] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
                                    10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                    20, 21, 22, 23, 24, 25, 26, 27, 28, 29};

  const int clsmap_4[gas_pcnst] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                   11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                   21, 22, 23, 24, 25, 26, 27, 28, 29, 30};

  std::vector<Real> lin_jac = {
      -0.3876940583e-4, -0.4444443653e-5, 0.1771299913e-5, -0.3712088560e-4,
      0.2646782869e-4,  -0.3116930588e-4, 0.0000000000,    0.0000000000,
      0.0000000000,     0.0000000000,     0.0000000000,    0.0000000000,
      0.0000000000,     0.0000000000,     0.0000000000,    0.0000000000,
      0.0000000000,     0.0000000000,     0.0000000000,    0.0000000000,
      0.0000000000,     0.0000000000,     0.0000000000,    0.0000000000,
      0.0000000000,     0.0000000000,     0.0000000000,    0.0000000000,
      0.0000000000,     0.0000000000,     0.0000000000,    0.0000000000};

  std::vector<Real> lrxt = {0.0000000000,    0.5166088810e-14, 0.3419820148e-5,
                            0.1771299913e-5, 0.8695113196e-5,  0.9402954384e-5,
                            0.1307123830e-4};

  std::vector<Real> lhet = {
      0.0000000000, 0.3534958569e-4, 0.4444443653e-5, 0.3534958569e-4,
      0.0000000000, 0.0000000000,    0.0000000000,    0.0000000000,
      0.0000000000, 0.0000000000,    0.0000000000,    0.0000000000,
      0.0000000000, 0.0000000000,    0.0000000000,    0.0000000000,
      0.0000000000, 0.0000000000,    0.0000000000,    0.0000000000,
      0.0000000000, 0.0000000000,    0.0000000000,    0.0000000000,
      0.0000000000, 0.0000000000,    0.0000000000,    0.0000000000,
      0.0000000000, 0.0000000000,    0.0000000000};

  std::vector<Real> iter_invariant = {
      0.3540757898e+13, 0.1322925052e-17, 0.7600857521e+15, 0.7078370470e-16,
      0.1032767998e+13, 0.3162618100e-13, 0.6921437694e+13, 0.6258597416e-12,
      0.9416920102e-14, 0.1273836795e-15, 0.2485200214e-14, 0.1080994459e-18,
      0.1800909531e7,   0.9778844460e-15, 0.3052719355e-13, 0.3441745081e-18,
      0.1909915045e-22, 0.4470095034e7,   0.4058339613e-15, 0.1927095488e-13,
      0.3715469722e-14, 0.1162980074e-14, 0.7379871842e-14, 0.4628399603e-13,
      0.1421930842e-19, 0.2233050316e3,   0.1493511427e-14, 0.3416929371e-15,
      0.7539655251e-25, 0.1383049154e5};

  std::vector<Real> lsol = {
      0.3535379752e-7,  0.1088693646e-9,  0.4762530189e-14, 0.2736308593e-11,
      0.2548213369e-12, 0.3549685501e-10, 0.1138542516e-9,  0.2491717570e-9,
      0.2253095070e-8,  0.3390091237e-10, 0.4585812463e-12, 0.8946720770e-11,
      0.3891580054e-15, 0.6483274310e10,  0.3520384006e-11, 0.1098978968e-9,
      0.1239028229e-14, 0.6875694162e-19, 0.1609234212e11,  0.1461002261e-11,
      0.6937543758e-10, 0.1337569100e-10, 0.4186728267e-11, 0.2656753863e-10,
      0.1666223857e-9,  0.5118951031e-16, 0.8038981136e6,   0.5376636899e-11,
      0.1230094141e-11, 0.2714275891e-21, 0.4978972675e8};
  std::vector<Real> solution = {
      0.1088693646e-9,  0.4762530189e-14, 0.2736308593e-11, 0.2548213369e-12,
      0.3549685501e-10, 0.1138542516e-9,  0.2491717570e-9,  0.2253095070e-8,
      0.3390091237e-10, 0.4585812463e-12, 0.8946720770e-11, 0.3891580054e-15,
      0.6483274310e10,  0.3520384006e-11, 0.1098978968e-9,  0.1239028229e-14,
      0.6875694162e-19, 0.1609234212e11,  0.1461002261e-11, 0.6937543758e-10,
      0.1337569100e-10, 0.4186728267e-11, 0.2656753863e-10, 0.1666223857e-9,
      0.5118951031e-16, 0.8038981136e6,   0.5376636899e-11, 0.1230094141e-11,
      0.2714275891e-21, 0.4978972675e8};

  newton_raphson_iter(dti, lin_jac.data(), lrxt.data(), lhet.data(),
                      iter_invariant.data(), factor, permute_4, clsmap_4,
                      lsol.data(), solution.data(), converged, convergence,
                      prod.data(), loss.data(), max_delta.data(), epsilon);

  int conv_count = 0;
  for (int i = 0; i < clscnt4; ++i) {
    if (!converged[i]) {
      conv_count += 1;
      logger.debug("Did not converge for i = {} and max_delta[{}] = {}", i, i,
                   max_delta[i]);
    }
  }
  logger.debug("{} entries failed to converge", conv_count);
  REQUIRE(conv_count == clscnt4);
}

TEST_CASE("test_solver_4_partial_failure", "mam4_gaschem_unit_tests") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger("gaschem unit tests",
                                ekat::logger::LogLevel::debug, comm);
  logger.debug("====================================================");
  logger.debug("\n ***gas chem unit tests Case 3: partial failure***");
  logger.debug("====================================================");
  const Real zero = 0;
  const Real dti = 0.2777777778e-3;
  const Real eps_val = 1.0e-17;

  logger.debug("case 3 error tolerance = {}", eps_val);

  std::vector<Real> prod(clscnt4, zero);
  std::vector<Real> loss(clscnt4, zero);
  std::vector<Real> max_delta(clscnt4, zero);

  Real epsilon[clscnt4];
  bool factor[itermax];
  bool converged[clscnt4];
  bool convergence = false;

  for (int i = 0; i < clscnt4; ++i) {
    epsilon[i] = eps_val;
    converged[i] = true;
  }
  for (int i = 0; i < itermax; ++i) {
    factor[i] = true;
  }

  const int gas_pcnst = 31; // number of gas phase species
  const int clscnt4 = 30;   // number of species in implicit class
  const int permute_4[gas_pcnst] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
                                    10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                    20, 21, 22, 23, 24, 25, 26, 27, 28, 29};

  const int clsmap_4[gas_pcnst] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                   11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                   21, 22, 23, 24, 25, 26, 27, 28, 29, 30};

  std::vector<Real> lin_jac = {
      -0.3876940583e-4, -0.4444443653e-5, 0.1771299913e-5, -0.3712088560e-4,
      0.2646782869e-4,  -0.3116930588e-4, 0.0000000000,    0.0000000000,
      0.0000000000,     0.0000000000,     0.0000000000,    0.0000000000,
      0.0000000000,     0.0000000000,     0.0000000000,    0.0000000000,
      0.0000000000,     0.0000000000,     0.0000000000,    0.0000000000,
      0.0000000000,     0.0000000000,     0.0000000000,    0.0000000000,
      0.0000000000,     0.0000000000,     0.0000000000,    0.0000000000,
      0.0000000000,     0.0000000000,     0.0000000000,    0.0000000000};

  std::vector<Real> lrxt = {0.0000000000,    0.5166088810e-14, 0.3419820148e-5,
                            0.1771299913e-5, 0.8695113196e-5,  0.9402954384e-5,
                            0.1307123830e-4};

  std::vector<Real> lhet = {
      0.0000000000, 0.3534958569e-4, 0.4444443653e-5, 0.3534958569e-4,
      0.0000000000, 0.0000000000,    0.0000000000,    0.0000000000,
      0.0000000000, 0.0000000000,    0.0000000000,    0.0000000000,
      0.0000000000, 0.0000000000,    0.0000000000,    0.0000000000,
      0.0000000000, 0.0000000000,    0.0000000000,    0.0000000000,
      0.0000000000, 0.0000000000,    0.0000000000,    0.0000000000,
      0.0000000000, 0.0000000000,    0.0000000000,    0.0000000000,
      0.0000000000, 0.0000000000,    0.0000000000};

  std::vector<Real> iter_invariant = {
      0.3540757898e+13, 0.1322925052e-17, 0.7600857521e+15, 0.7078370470e-16,
      0.1032767998e+13, 0.3162618100e-13, 0.6921437694e+13, 0.6258597416e-12,
      0.9416920102e-14, 0.1273836795e-15, 0.2485200214e-14, 0.1080994459e-18,
      0.1800909531e7,   0.9778844460e-15, 0.3052719355e-13, 0.3441745081e-18,
      0.1909915045e-22, 0.4470095034e7,   0.4058339613e-15, 0.1927095488e-13,
      0.3715469722e-14, 0.1162980074e-14, 0.7379871842e-14, 0.4628399603e-13,
      0.1421930842e-19, 0.2233050316e3,   0.1493511427e-14, 0.3416929371e-15,
      0.7539655251e-25, 0.1383049154e5};

  std::vector<Real> lsol = {
      0.3535379752e-7,  0.1088693646e-9,  0.4762530189e-14, 0.2736308593e-11,
      0.2548213369e-12, 0.3549685501e-10, 0.1138542516e-9,  0.2491717570e-9,
      0.2253095070e-8,  0.3390091237e-10, 0.4585812463e-12, 0.8946720770e-11,
      0.3891580054e-15, 0.6483274310e10,  0.3520384006e-11, 0.1098978968e-9,
      0.1239028229e-14, 0.6875694162e-19, 0.1609234212e11,  0.1461002261e-11,
      0.6937543758e-10, 0.1337569100e-10, 0.4186728267e-11, 0.2656753863e-10,
      0.1666223857e-9,  0.5118951031e-16, 0.8038981136e6,   0.5376636899e-11,
      0.1230094141e-11, 0.2714275891e-21, 0.4978972675e8};
  std::vector<Real> solution = {
      0.1088693646e-9,  0.4762530189e-14, 0.2736308593e-11, 0.2548213369e-12,
      0.3549685501e-10, 0.1138542516e-9,  0.2491717570e-9,  0.2253095070e-8,
      0.3390091237e-10, 0.4585812463e-12, 0.8946720770e-11, 0.3891580054e-15,
      0.6483274310e10,  0.3520384006e-11, 0.1098978968e-9,  0.1239028229e-14,
      0.6875694162e-19, 0.1609234212e11,  0.1461002261e-11, 0.6937543758e-10,
      0.1337569100e-10, 0.4186728267e-11, 0.2656753863e-10, 0.1666223857e-9,
      0.5118951031e-16, 0.8038981136e6,   0.5376636899e-11, 0.1230094141e-11,
      0.2714275891e-21, 0.4978972675e8};

  newton_raphson_iter(dti, lin_jac.data(), lrxt.data(), lhet.data(),
                      iter_invariant.data(), factor, permute_4, clsmap_4,
                      lsol.data(), solution.data(), converged, convergence,
                      prod.data(), loss.data(), max_delta.data(), epsilon);

  int conv_count = 0;
  for (int i = 0; i < clscnt4; ++i) {
    if (!converged[i]) {
      conv_count += 1;
      logger.debug("Did not converge for i = {} and max_delta[{}] = {}", i, i,
                   max_delta[i]);
    }
  }
  logger.debug("{} entries failed to converge", conv_count);
  REQUIRE(conv_count > 0);
}
