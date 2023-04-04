// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_AERO_MODEL_OD_HPP
#define MAM4XX_AERO_MODEL_OD_HPP

#include <ekat/util/ekat_math_utils.hpp>

#include <haero/atmosphere.hpp>
#include <haero/math.hpp>

#include <mam4xx/aero_config.hpp>
#include <mam4xx/conversions.hpp>
#include <mam4xx/mam4_types.hpp>
#include <mam4xx/utils.hpp>

namespace mam4 {


namespace aero_model_od {

} // end namespace aero_model_od

/// @class aero_model_od
/// This class implements MAM4's aero_model_od parameterization.
class AeroModelOD {
public:
	// aero_model_od-specific configuration
  struct Config {
  	Config() {}
  	Config(const Config &) = default;
    ~Config() = default;
    Config &operator=(const Config &) = default;
  };

  private:
  Config config_;
  public:
  // name--unique name of the process implemented by this class
  const char *name() const { return "MAM4 aero_model_od"; }
};  // end class aero_model_od

} // end namespace mam4

#endif
