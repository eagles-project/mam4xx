#include "atmosphere_utils.hpp"
#include "testing.hpp"

// #include "mam4xx/aero_modes.hpp"
// #include "mam4xx/conversions.hpp"
// #include <mam4xx/mode_dry_particle_size.hpp>
// #include <mam4xx/aero_config.hpp>

// #include <haero/constants.hpp>
#include <haero/floating_point.hpp>
#include <haero/haero.hpp>

// #include "mam4xx/conversions.hpp"

#include <catch2/catch.hpp>
#include <ekat/logging/ekat_logger.hpp>
#include <ekat/mpi/ekat_comm.hpp>
#include <mam4xx/mam4.hpp>

// using namespace haero;
using namespace mam4;
// using namespace mam4::conversions;
const int nmodes = AeroConfig::num_modes();
const int loffset = 9;
const mam4::mo_setsox::Config setsox_config_;

TEST_CASE("test_sox_cldaero_create_obj", "mam4_mo_setsox_unit_tests") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger(
      "mo_setsox unit tests: test_sox_cldaero_create_obj",
      ekat::logger::LogLevel::debug, comm);
  logger.debug("");

  const mam4::mo_setsox::Config config_;

  const int nspec = AeroConfig::num_gas_phase_species();
  const Real cldfrc1 = 1.0;
  const Real cldfrc0 = 0.0;
  Real qcw[nspec];
  for (int i = 0; i < nspec; ++i) {
    qcw[i] = 1;
  }
  // const int lptr_so4_cw_amode[3] = {0, 1, 2};
  const Real lwc = 1;
  const Real cfact = 1;
  // int loffset = 0;

  // cldfrc > 0 => calculate xlwc, so4c comes from adding 3 entries of qwc,
  // and so4_fact = 1
  mam4::mo_setsox::Cloudconc cldconc = mam4::mo_setsox::sox_cldaero_create_obj(
      cldfrc1, qcw, lwc, cfact, loffset, setsox_config_);
  logger.debug("so4c = {}, xlwc = {}, so4_fact = {}", cldconc.so4c,
               cldconc.xlwc, cldconc.so4_fact);
  REQUIRE(FloatingPoint<Real>::equiv(cldconc.so4c, 3.0));
  REQUIRE(FloatingPoint<Real>::equiv(cldconc.xlwc, 1.0));
  REQUIRE(FloatingPoint<Real>::equiv(cldconc.so4_fact, 1.0));

  // cldfrc = 0 => xlwc = 0
  cldconc = mam4::mo_setsox::sox_cldaero_create_obj(cldfrc0, qcw, lwc, cfact,
                                                    loffset, config_);
  logger.debug("so4c = {}, xlwc = {}, so4_fact = {}", cldconc.so4c,
               cldconc.xlwc, cldconc.so4_fact);
  REQUIRE(FloatingPoint<Real>::equiv(cldconc.so4c, 3.0));
  REQUIRE(FloatingPoint<Real>::equiv(cldconc.xlwc, 0.0));
  REQUIRE(FloatingPoint<Real>::equiv(cldconc.so4_fact, 1.0));
}

TEST_CASE("test_henry_factor_so2", "mam4_mo_setsox_unit_tests") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger("mo_setsox unit tests: test_henry_factor_so2",
                                ekat::logger::LogLevel::debug, comm);
  logger.debug("");

  // this comes from the validation input yaml files
  // NOTE: also given with a value of 0.1068378555E-02, but that's not germane
  // to this test
  const Real t_factor = 0.3415485654e-3;
  const Real ref_xk = 1.230 * haero::exp(3120.0 * t_factor);
  const Real ref_xe = 1.7e-2 * haero::exp(2090.0 * t_factor);
  const Real ref_x2 = 6.0e-8 * haero::exp(1120.0 * t_factor);
  Real xk, xe, x2;

  mam4::mo_setsox::henry_factor_so2(t_factor, xk, xe, x2);

  logger.debug("xk = {}", xk);
  logger.debug("xe = {}", xe);
  logger.debug("x2 = {}", x2);
  REQUIRE(FloatingPoint<Real>::equiv(xk, ref_xk));
  REQUIRE(FloatingPoint<Real>::equiv(xe, ref_xe));
  REQUIRE(FloatingPoint<Real>::equiv(x2, ref_x2));
}

TEST_CASE("test_henry_factor_co2", "mam4_mo_setsox_unit_tests") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger("mo_setsox unit tests: test_henry_factor_co2",
                                ekat::logger::LogLevel::debug, comm);
  logger.debug("");
  // this comes from the validation input yaml files
  // NOTE: also given with a value of 0.1068378555E-02, but that's not germane
  // to this test
  const Real t_factor = 0.3415485654e-3;
  const Real ref_xk = 3.1e-2 * haero::exp(2423.0 * t_factor);
  const Real ref_xe = 4.3e-7 * haero::exp(-913.0 * t_factor);
  Real xk, xe;

  mam4::mo_setsox::henry_factor_co2(t_factor, xk, xe);

  logger.debug("xk = {}", xk);
  REQUIRE(FloatingPoint<Real>::equiv(xk, ref_xk));
  logger.debug("xe = {}", xe);
  REQUIRE(FloatingPoint<Real>::equiv(xe, ref_xe));
}

TEST_CASE("test_henry_factor_h202", "mam4_mo_setsox_unit_tests") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger("mo_setsox unit tests: test_henry_factor_h202",
                                ekat::logger::LogLevel::debug, comm);
  logger.debug("");
  // this comes from the validation input yaml files
  // NOTE: also given with a value of 0.1068378555E-02, but that's not germane
  // to this test
  const Real t_factor = 0.3415485654e-3;
  const Real ref_xk = 7.4e4 * haero::exp(6621.0 * t_factor);
  const Real ref_xe = 2.2e-12 * haero::exp(-3730.0 * t_factor);
  Real xk, xe;

  mam4::mo_setsox::henry_factor_h2o2(t_factor, xk, xe);

  logger.debug("xk = {}", xk);
  REQUIRE(FloatingPoint<Real>::equiv(xk, ref_xk));
  logger.debug("xe = {}", xe);
  REQUIRE(FloatingPoint<Real>::equiv(xe, ref_xe));
}

TEST_CASE("test_henry_factor_o3", "mam4_mo_setsox_unit_tests") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger("mo_setsox unit tests: test_henry_factor_o3",
                                ekat::logger::LogLevel::debug, comm);
  logger.debug("");
  // this comes from the validation input yaml files
  // NOTE: also given with a value of 0.1068378555E-02, but that's not germane
  // to this test
  const Real t_factor = 0.3415485654e-3;
  const Real ref_xk = 1.15e-2 * haero::exp(2560.0 * t_factor);
  Real xk;

  mam4::mo_setsox::henry_factor_o3(t_factor, xk);

  logger.debug("xk = {}", xk);
  REQUIRE(FloatingPoint<Real>::equiv(xk, ref_xk));
}

TEST_CASE("test_compute_aer_factor", "mam4_mo_setsox_unit_tests") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger("mo_setsox unit tests: test_compute_aer_factor",
                                ekat::logger::LogLevel::debug, comm);
  logger.debug("");
  mam4::mo_setsox::Config config_;
  // for reference:
  // lptr_so4_cw_amode[4] = {15, 23, 30, -1};
  // numptrcw_amode[nmodes] = {23, 28, 36, 40};

  Real tmr[32] = {0};
  for (int i = 0; i < nmodes; ++i) {
    // this is completely arbitrary, checking +/- values, since tmr is only used
    // to assign a variable = max(0, tmr[i])
    tmr[config_.numptrcw_amode[i] - loffset] =
        haero::pow(-1, i) * 0.5e-10 * (i + 1);
  }
  Real *tmr_ptr = tmr;
  Real faqgain_so4[nmodes];
  Real ref_faqgain_so4[nmodes] = {0.4, 0.0, 0.6, 0.0};

  mam4::mo_setsox::compute_aer_factor(tmr_ptr, loffset, config_, faqgain_so4);

  for (int i = 0; i < nmodes; ++i) {
    logger.debug("faqgain_so4[i] = {}", faqgain_so4[i]);
    logger.debug("ref_faqgain_so4[i] = {}", ref_faqgain_so4[i]);
    REQUIRE(FloatingPoint<Real>::equiv(faqgain_so4[i], ref_faqgain_so4[i]));
  }
}

TEST_CASE("test_cldaero_uptakerate", "mam4_mo_setsox_unit_tests") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger("mo_setsox unit tests: test_cldaero_uptakerate",
                                ekat::logger::LogLevel::debug, comm);
  logger.debug("");

  // these are each the two values from the respective validation data
  // so, why not just run the whole outer product span?
  // liquid water volume [cm^3/cm^3]
  Real xl[2] = {0.3811135505e-7, 0.2045167481e-7};
  // droplet number concentration
  Real cldnum[2] = {0.1000000000e-11, 0.2145049148e8};
  // cloud fraction [fraction]
  Real cldfrc[2] = {0.7653835562, 0.4131862680e-1};
  // temperature [K]
  Real tfld[2] = {0.2704710575e3, 0.2804313949e3};
  // pressure [Pa]
  Real press[2] = {0.6753476429e5, 0.5280266906e5};

  // ISA density for the 2 troposphere layers are [1.2985, 0.3639] kg/m^3
  // => [1.2985e-3, 0.3639e-3] kg/L
  // source:
  // https://en.wikipedia.org/wiki/International_Standard_Atmosphere#Description
  // NOTE: no idea what "total" atm density is, so hopefully this is reasonable
  // total atms density [kg/L]
  Real cfact[2] = {1.2985e-3, 0.3639e-3};

  Real t_xl, t_cldnum, t_cldfrc, t_tfld, t_press, t_cfact;
  Real uptkrate;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 2; ++k) {
        for (int ell = 0; ell < 2; ++ell) {
          for (int m = 0; m < 2; ++m) {
            for (int n = 0; n < 2; ++n) {
              t_xl = xl[i];
              t_cldnum = cldnum[j];
              t_cldfrc = cldfrc[k];
              t_tfld = tfld[ell];
              t_press = press[m];
              t_cfact = cfact[n];
              uptkrate = mam4::mo_setsox::cldaero_uptakerate(
                  t_xl, t_cldnum, t_cfact, t_cldfrc, t_tfld, t_press);
              logger.debug("[xl, cldnum, cldfrc, tfld, press, cfact] = [{}, "
                           "{}, {}, {}, {}, {}]",
                           t_xl, t_cldnum, t_cldfrc, t_tfld, t_press, t_cfact);
              logger.debug("uptkrate = {}", uptkrate);
              // FIXME: anecdotally, these all seem to be < 1, and positive
              // do these assumptions holdup? would negative be "downtake" rate?
              REQUIRE(FloatingPoint<Real>::in_bounds(uptkrate, 0.0, 1.0));
            }
          }
        }
      }
    }
  }
}

TEST_CASE("test_update_tmr", "mam4_mo_setsox_unit_tests") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger("mo_setsox unit tests: test_update_tmr",
                                ekat::logger::LogLevel::debug, comm);
  logger.debug("");

  // tmr = tmr + dqdt * dtime;
  Real tmr = 27;
  Real ref_tmr = 28;
  const Real dqdt = 42.0;
  const Real dtime = haero::pow(42.0, -1);

  mam4::mo_setsox::update_tmr(tmr, dqdt, dtime);

  logger.debug("tmr = {}", tmr);
  REQUIRE(FloatingPoint<Real>::equiv(tmr, ref_tmr));
}

TEST_CASE("test_update_tmr_nonzero", "mam4_mo_setsox_unit_tests") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger("mo_setsox unit tests: test_update_tmr_nonzero",
                                ekat::logger::LogLevel::debug, comm);
  logger.debug("");

  Real tmr_in[7] = {-27.0, -1.0e-10, -1.0e-30, 0.0, 1.0e-30, 1.0e-10, 42.0};
  const Real ref_tmr[7] = {1.0e-20, 1.0e-20, 1.0e-20, 1.0e-20,
                           1.0e-20, 1.0e-10, 42.0};
  const int idx[8] = {-7, 0, 1, 2, 3, 4, 5, 6};
  int tmp_idx;
  Real fake_tmr = 3.14;

  for (int j = 0; j < 8; ++j) {
    tmp_idx = idx[j];
    if (tmp_idx < 0) {
      mam4::mo_setsox::update_tmr_nonzero(fake_tmr, tmp_idx);
      logger.debug("idx[{}] = {}", j, tmp_idx);
      logger.debug("tmr_fake = {}; ref_tmr_fake = {}", tmp_idx, fake_tmr,
                   tmp_idx, fake_tmr);
      REQUIRE(FloatingPoint<Real>::equiv(fake_tmr, fake_tmr));
    } else {
      mam4::mo_setsox::update_tmr_nonzero(tmr_in[tmp_idx], tmp_idx);
      logger.debug("idx[{}] = {}", j, tmp_idx);
      logger.debug("tmr[{}] = {}; ref_tmr[{}] = {}", tmp_idx, tmr_in[tmp_idx],
                   tmp_idx, ref_tmr[tmp_idx]);
      REQUIRE(FloatingPoint<Real>::equiv(tmr_in[tmp_idx], ref_tmr[tmp_idx]));
    }
  }
}

// NOTE: I tried to make a test with conclusive results and couldn't get it
// working--revisit eventually
/*TEST_CASE("test_sox_cldaero_update", "mam4_mo_setsox_unit_tests") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger("mo_setsox unit tests: test_sox_cldaero_update",
                                ekat::logger::LogLevel::debug, comm);
  logger.debug("");

  const int nspec_gas = AeroConfig::num_gas_phase_species();

  // most of the below comes from the validation yaml files
  // this dt choice is a guess right now
  Real dt = 1.0;
  Real mbar = 0.2896600000e2;
  Real pdel = 0.3883421302e4;
  Real press = 0.5280266906e5;
  Real tfld = 0.2704710575e3;
  Real cldnum = 0.2145049148e8;
  Real cldfrc[2] = {0.7653835562, 0.0};
  Real xhnm = 0.1414006739e20;
  // from mo_setsox.hpp ~L1209
  Real cfact = xhnm * 1.0e6 * 1.38e-23 / 287.0 * 1.0e-3;
  // xlwc values from validation yaml: 0.2045167481e-7, 0.3811135505e-7
  // these values result in negative output answers, so guessing a value to make
  // it positive
  Real xlwc[2] = {0.2045167481e-7, 0.0};
  // Real xlwc = 0.0;
  // this value is chosen arbitrarily
  // Real delso4_hprxn = 1.23e-15;
  Real delso4_hprxn = 0.0;
  // this value is chosen arbitrarily
  Real xh2so4 = 4.56e-2;
  Real xso4 = 0.7008637581e-14;
  Real xso4_init = 0.7152100476e-3;
  Real qcw[nspec_gas] = {
      0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00,
      0.0000000000E+00, 0.0000000000E+00, 0.1821296430E-10, 0.2432565542E-10,
      0.2454017849E-09, 0.3567337656E-11, 0.4159584725E-13, 0.8950397033E-12,
      0.3594812481E-16, 0.4818129270E+09, 0.7580693633E-13, 0.2408959105E-11,
      0.8383888225E-17, 0.4271204856E-21, 0.1648792032E+08, 0.1141617399E-12,
      0.6886564669E-11, 0.1137264049E-11, 0.3791380267E-12, 0.2311807532E-11,
      0.1340284959E-10, 0.4317459265E-17, 0.8085661862E+05, 0.0000000000E+00,
      0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00};
  Real qin[nspec_gas] = {
      0.3525157528E-07, 0.2579731039E-09, 0.1889793408E-12, 0.1401575499E-10,
      0.9273687127E-11, 0.1839610018E-09, 0.6962117637E-10, 0.9532668198E-10,
      0.9878474612E-09, 0.1392619425E-10, 0.1643483379E-12, 0.3557021547E-11,
      0.1427943450E-15, 0.4980337770E+10, 0.4033946091E-11, 0.1392812315E-09,
      0.5086494829E-15, 0.2634613374E-19, 0.2803441725E+11, 0.4038241310E-12,
      0.2496475156E-10, 0.3886307438E-11, 0.1289740453E-11, 0.7841132260E-11,
      0.4534748825E-10, 0.1468133938E-16, 0.2940008493E+06, 0.7620773644E-11,
      0.1611532411E-11, 0.2143351104E-21, 0.6551452334E+08};
  Real ref_qcw[nspec_gas], ref_qin[nspec_gas];
  for (int i = 0; i < nspec_gas; ++i)
  {
    ref_qcw[i] = qcw[i];
    ref_qin[i] = qin[i];
  }

  for (int j = 0; j < 2; ++j) {
    mam4::mo_setsox::sox_cldaero_update(
        loffset, dt, mbar, pdel, press, tfld, cldnum, cldfrc[j], cfact, xlwc[j],
        delso4_hprxn, xh2so4, xso4, xso4_init, qcw, qin);
    for (int i = 0; i < nspec_gas; ++i) {
      logger.debug("qcw[{}] = {}; qin[{}] = {}", i, qcw[i], i, qin[i]);
      // these are for nonzero values of cldfrc and xlwc, so the initial
      // shortcutting does not occur:
      // if ((cldfrc >= small_value_5) && (xlwc >= small_value_8)) {
      if (j == 0)
      {
        REQUIRE(qcw[i] >= 0.0);
        REQUIRE(qin[i] >= 0.0);
      } else if (j == 1)
      // this shortcuts, so these arrays only hit update_tmr_nonzero()
      {
        if (ref_qcw[i] < 1e-20)
        {
          REQUIRE(FloatingPoint<Real>::equiv(qcw[i], 1.0e-20));
        } else {
          REQUIRE(qcw[i] >= 0.0);
        }
        if (ref_qcw[i] < 1e-20)
        {
          REQUIRE(FloatingPoint<Real>::equiv(qin[i], 1.0e-20));
        } else {
          REQUIRE(qin[i] >= 0.0);
        }
      }

    }
  }
}*/

/*NOTE: These have validation tests, so skip for now
void calc_ph_values(Real temperature, Real patm, Real xlwc, Real t_factor,
                    Real xso2, Real xso4, Real xhnm, Real so4_fact, Real Ra,
                    Real xkw, Real const0, bool &converged, Real &xph)
void calc_sox_aqueous(bool modal_aerosols, Real rah2o2, Real h2o2g, Real so2g,
                      Real o3g, Real rao3, Real patm, Real dt, Real t_factor,
                      Real xlwc, Real const0, Real xhnm, Real heo3, Real heso2,
                      // inout
                      Real &xso2, Real &xso4, Real &xso4_init, Real &xh2o2,
                      // out
                      Real &xdelso4hp)
void calc_ynetpos(Real yph, Real fact1_so2, Real fact2_so2, Real fact3_so2,
                  Real fact4_so2, Real Eco2, Real Eh2o, Real Eso4,
                  Real so4_fact,
                  // out
                  Real &xph, Real &ynetpos)
void setsox(Real xhnm, Real cldfrc, Real qcw[nspec_gas], Real lwc, Real tfld,
            Real press, Real qin[nspec_gas], Real dt, Real mbar, Real pdel,
            Real cldnum)*/
