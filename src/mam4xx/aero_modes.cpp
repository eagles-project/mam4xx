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

std::string aero_id_str(const AeroId aid) {
  switch (aid) {
  case (AeroId::SO4): {
    return "sulphate";
  }
  case (AeroId::POM): {
    return "primary_organic_matter";
  }
  case (AeroId::SOA): {
    return "secondary_organic_aerosol";
  }
  case (AeroId::BC): {
    return "black_carbon";
  }
  case (AeroId::DST): {
    return "dust";
  }
  case (AeroId::NaCl): {
    return "salt";
  }
  case (AeroId::MOM): {
    return "marine_organic_matter";
  }
  case (AeroId::None): {
    return "none";
  }
  default:
    return "invalid_aerosol_id";
  }
}

} // namespace mam4
