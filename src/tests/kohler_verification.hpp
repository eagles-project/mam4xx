// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_KOHLER_VERIFICATION_HPP
#define MAM4XX_KOHLER_VERIFICATION_HPP

#include <haero/haero.hpp>

#ifdef HAERO_DOUBLE_PRECISION
#include <mam4xx/kohler.hpp>

namespace mam4 {

struct KohlerVerification {
  int n;
  int n_trials;
  DeviceType::view_1d<Real> true_sol;
  DeviceType::view_1d<Real> relative_humidity;
  DeviceType::view_1d<Real> hygroscopicity;
  DeviceType::view_1d<Real> dry_radius;
  Real rhmin;
  Real drh;
  Real hmin;
  Real dhyg;
  Real rmin;
  Real ddry;

  /** Constructor

    @param [in] n number of trials for each of the 3 parameters
  */
  explicit KohlerVerification(const int nn)
      : n(nn), n_trials(haero::cube(nn)),
        true_sol("kohler_true_sol", haero::cube(nn)),
        relative_humidity("relative_humidity", haero::cube(nn)),
        hygroscopicity("hygroscopicity", haero::cube(nn)),
        dry_radius("dry_radius", haero::cube(nn)),
        rhmin(KohlerPolynomial::rel_humidity_min),
        drh((KohlerPolynomial::rel_humidity_max -
             KohlerPolynomial::rel_humidity_min) /
            (nn - 1)),
        hmin(KohlerPolynomial::hygro_min),
        dhyg((KohlerPolynomial::hygro_max - KohlerPolynomial::hygro_min) /
             (nn - 1)),
        rmin(KohlerPolynomial::dry_radius_min_microns),
        ddry((KohlerPolynomial::dry_radius_max_microns -
              KohlerPolynomial::dry_radius_min_microns) /
             (nn - 1)) {
    generate_input_data();
    load_true_sol_from_file();
  }

  /**
    builds an array of n input values for each
    parameter in the Kohler polynomial for use on device,
    for a total of n**3 trials.
  */
  void generate_input_data();

  /** @brief Writes a string containing a Mathematica script that may be used to
    generate the verification data.

    @param [in] n the number of points in each parameter range
  */
  std::string mathematica_verification_program() const;

  /** @brief Writes a string containing a Matlab script that may be used to
    generate the verification data.

    @param [in] n the number of points in each parameter range
  */
  std::string matlab_verification_program() const;

  /** This function loads verification data from a text file into a view.
   */
  void load_true_sol_from_file();
};

} // namespace mam4
#endif // double precision
#endif
