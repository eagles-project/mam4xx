#include <mam4xx/calcsize.hpp>
#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;

void update_diameter_and_vol2num(Ensemble *ensemble) {

  // We don't need any settings for this particular test.
  // Settings settings = ensemble->settings();

  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    // Fetch ensemble parameters

    // drv and num can be either interstitial or cloudborne aerosols
    const Real drv = input.get("drv");
    const Real num = input.get("num");

    // mode dependent
    Real v2nmin = input.get("v2nmin");
    Real v2nmax = input.get("v2nmax");

    Real dgnmin = input.get("dgnmin");
    Real dgnmax = input.get("dgnmax");

    Real cmn_factor = input.get("cmn_factor");

    Real dgncur_k_i = 0;
    Real v2ncur_k_i = 0;

    // Kokkos::parallel_for(
    //     "update_diameter_and_vol2num", 1, [&] KOKKOS_FUNCTION(int i) {
    calcsize::update_diameter_and_vol2num(drv, num, v2nmin, v2nmax, dgnmin,
                                          dgnmax, cmn_factor, dgncur_k_i,
                                          v2ncur_k_i);
    //    });

    output.set("dgncur", dgncur_k_i);
    output.set("v2ncur", v2ncur_k_i);
  });
}
