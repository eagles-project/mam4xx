// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include "aero_species.hpp"

namespace mam4 {

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

std::string aero_id_short_name(const AeroId aid) {
  switch (aid) {
  case (AeroId::SO4): {
    return "so4";
  }
  case (AeroId::POM): {
    return "pom";
  }
  case (AeroId::SOA): {
    return "soa";
  }
  case (AeroId::BC): {
    return "bc";
  }
  case (AeroId::DST): {
    return "dst";
  }
  case (AeroId::NaCl): {
    return "nacl";
  }
  case (AeroId::MOM): {
    return "mom";
  }
  case (AeroId::None): {
    return "none";
  }
  default:
    return "invalid_aerosol_id";
  }
}

} // namespace mam4
