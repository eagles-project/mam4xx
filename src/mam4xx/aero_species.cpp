#include "mam4xx/aero_species.hpp"

namespace mam4 {

void set_aero_molecular_weight(const AeroId id, Real molecular_weight) {
  aero_species_[int(id)].molecular_weight = molecular_weight;
}

void set_aero_density(const AeroId id, Real density) {
  aero_species_[int(id)].density = density;
}

void set_aero_hygroscopicity(const AeroId id, Real hygroscopicity) {
  aero_species_[int(id)].hygroscopicity = hygroscopicity;
}

std::string aero_id_str(const AeroId id) {
  switch (id) {
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

std::string aero_id_short_name(const AeroId id) {
  switch (id) {
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
