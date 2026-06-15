// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include "aero_modes.hpp"
#include <string>

namespace mam4 {

/// Map ModeIndex to string (for logging, e.g.)
std::string mode_str(const ModeIndex m) {
  switch (m) {
  case (ModeIndex::Accumulation): {
    return "accumulation";
    break;
  }
  case (ModeIndex::Aitken): {
    return "aitken";
    break;
  }
  case (ModeIndex::Coarse): {
    return "coarse";
    break;
  }
  case (ModeIndex::PrimaryCarbon): {
    return "primary_carbon";
    break;
  }
  default: {
    return "invalid_mode_index";
  }
  }
}

} // namespace mam4
