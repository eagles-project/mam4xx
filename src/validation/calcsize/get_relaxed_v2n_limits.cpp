#include <haero/constants.hpp>
#include <iostream>
#include <mam4xx/calcsize.hpp>
#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;

void get_relaxed_v2n_limits(Ensemble* ensemble) {

  // We don't need any settings for this particular test.
  // Settings settings = ensemble->settings();

  // Run the ensemble.
  ensemble->process([=](const Input& input, Output& output) {
    // Fetch ensemble parameters

    const bool  do_aitacc_transfer = static_cast<bool>(input.get("do_aitacc_transfer"));
    const bool  is_aitken_mode = static_cast<bool>(input.get("is_aitken_mode"));
    const bool  is_accum_mode = static_cast<bool>(input.get("is_accum_mode"));

    Real v2nmin = input.get("v2nmin");
    Real v2nmax = input.get("v2nmax");
    Real v2nminrl = 0.0;
    Real v2nmaxrl =0.0;
 
    // Call the cluster growth function on device.
    Kokkos::parallel_for(
        "get_relaxed_v2n_limits", 1, [&] KOKKOS_FUNCTION(int i) {
          calcsize::get_relaxed_v2n_limits(do_aitacc_transfer,
                                          is_aitken_mode,
                                          is_accum_mode, v2nmin,
                                          v2nmax, v2nminrl, v2nmaxrl); 
        });

    output.set("v2nmin", v2nmin);
    output.set("v2nmax", v2nmax);
    output.set("v2nminrl", v2nminrl);
    output.set("v2nmaxrl", v2nmaxrl);
    
  });
}
