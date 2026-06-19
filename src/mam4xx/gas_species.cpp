// Copyright (c) 2021, National Technology & Engineering Solutions of Sandia,
// LLC (NTESS). Copyright (c) 2022, Battelle Memorial Institute
// SPDX-License-Identifier: BSD-3-Clause

#include "gas_species.hpp"
#include "mam4_constants.hpp"

namespace mam4 {

GasSpeciesHostView default_gas_species() {
  GasSpeciesHostView species("Gas species", int(GasId::NumSpecies));
  species[int(GasId::O3)] = GasSpecies{defaults::molec_weight_o3};
  species[int(GasId::H2O2)] = GasSpecies{defaults::molec_weight_h2o2};
  species[int(GasId::H2SO4)] = GasSpecies{Constants::molec_weight_h2so4};
  species[int(GasId::SO2)] = GasSpecies{defaults::molec_weight_so2};
  species[int(GasId::DMS)] = GasSpecies{defaults::molec_weight_dms};
  species[int(GasId::SOAG)] = GasSpecies{Constants::molec_weight_c};
  species[int(GasId::O2)] = GasSpecies{defaults::molec_weight_o2};
  species[int(GasId::CO2)] = GasSpecies{defaults::molec_weight_co2};
  species[int(GasId::N2O)] = GasSpecies{defaults::molec_weight_n2o};
  species[int(GasId::CH4)] = GasSpecies{defaults::molec_weight_ch4};
  species[int(GasId::CCl3F)] = GasSpecies{defaults::molec_weight_ccl3f};
  species[int(GasId::CHCl2F)] = GasSpecies{defaults::molec_weight_chcl2f};
  species[int(GasId::NH3)] = GasSpecies{Constants::molec_weight_nh3};
  return species;
}

GasSpeciesView
gas_species_on_device(const GasSpeciesHostView &species_on_host) {
  GasSpeciesView species_on_device("On-device gas species",
                                   species_on_host.extent(0));
  Kokkos::deep_copy(species_on_device, species_on_host);
  return species_on_device;
}

} // namespace mam4
