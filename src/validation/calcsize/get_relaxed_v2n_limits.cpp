#include <haero/constants.hpp>
#include <iostream>
#include <mam4xx/calcsize.hpp>
#include <mam4xx/mam4.hpp>
#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;

void get_relaxed_v2n_limits(Ensemble *ensemble) {

  // We don't need any settings for this particular test.
  // Settings settings = ensemble->settings();

  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    // Fetch ensemble parameters

    const bool do_aitacc_transfer =
        static_cast<bool>(input.get("do_aitacc_transfer"));
    const bool is_aitken_mode = static_cast<bool>(input.get("is_aitken_mode"));
    const bool is_accum_mode = static_cast<bool>(input.get("is_accum_mode"));

    mam4::AeroConfig mam4_config;
    mam4::CalcSizeProcess process(mam4_config);

    const auto nmodes = AeroConfig::num_modes();
    Real v2nmin_nmodes[nmodes];
    Real v2nmax_nmodes[nmodes];
    // should we get this values from calcsize config?
    for (int m = 0; m < nmodes; ++m) {
      const Real common_factor_nmodes =
          exp(4.5 * log(modes[m].mean_std_dev) * log(modes[m].mean_std_dev)) *
          Constants::pi_sixth; // A common factor
      v2nmin_nmodes[m] =
          1.0 / (common_factor_nmodes * pow(modes[m].max_diameter, 3.0));
      v2nmax_nmodes[m] =
          1.0 / (common_factor_nmodes * pow(modes[m].min_diameter, 3.0));
    }

    Real v2nminrl[nmodes] = {0.0};
    Real v2nmaxrl[nmodes] = {0.0};
    // Call the cluster growth function on device.
    Kokkos::parallel_for(
        "get_relaxed_v2n_limits", 1, [&] KOKKOS_FUNCTION(int k) {
          for (int m = 0; m < nmodes; ++m) {
            calcsize::get_relaxed_v2n_limits(
                do_aitacc_transfer, is_aitken_mode, is_accum_mode,
                v2nmin_nmodes[m], v2nmax_nmodes[m], v2nminrl[m], v2nmaxrl[m]);
          }
        });

    std::vector<Real> v2nminrl_values, v2nmaxrl_values;
    std::vector<Real> v2nmin_values, v2nmax_values;

    for (int m = 0; m < nmodes; ++m) {
      v2nminrl_values.push_back(v2nminrl[m]);
      v2nmaxrl_values.push_back(v2nmaxrl[m]);

      v2nmin_values.push_back(v2nmin_nmodes[m]);
      v2nmax_values.push_back(v2nmax_nmodes[m]);
    }

    output.set("v2nmin", v2nmin_values);
    output.set("v2nmax", v2nmax_values);
    output.set("v2nminrl", v2nminrl_values);
    output.set("v2nmaxrl", v2nmaxrl_values);
  });
}
