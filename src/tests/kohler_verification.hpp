#ifndef MAM4XX_VERIFICATION_HPP
#define MAM4XX_VERIFICATION_HPP

#include "kohler.hpp"

namespace mam4 {

struct KohlerVerification {
  int n;
  int n_trials;
  DeviceType::view_1d<PackType> true_sol;
  DeviceType::view_1d<PackType> relative_humidity;
  DeviceType::view_1d<PackType> hygroscopicity;
  DeviceType::view_1d<PackType> dry_radius;
  Real rhmin;
  Real drh;
  Real hmin;
  Real dhyg;
  Real rmin;
  Real ddry;

  /** Constructor

    @param [in] n number of trials for each of the 3 parameters
  */
  explicit KohlerVerification(const int nn) :
    n(nn),
    n_trials(haero::cube(nn)),
    true_sol("kohler_true_sol", haero::cube(nn)),
    relative_humidity("relative_humidity", haero::cube(nn)),
    hygroscopicity("hygroscopicity", haero::cube(nn)),
    dry_radius("dry_radius", haero::cube(nn)),
    rhmin(KohlerPolynomial<Real>::rel_humidity_min),
    drh((KohlerPolynomial<Real>::rel_humidity_max -
          KohlerPolynomial<Real>::rel_humidity_min) / (nn-1)),
    hmin(KohlerPolynomial<Real>::hygro_min),
    dhyg((KohlerPolynomial<Real>::hygro_max -
          KohlerPolynomial<Real>::hygro_min) / (nn-1)),
    rmin(KohlerPolynomial<Real>::dry_radius_min_microns),
    ddry((KohlerPolynomial<Real>::dry_radius_max_microns -
          KohlerPolynomial<Real>::dry_radius_min_microns) / (nn-1))
  {
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
#endif
