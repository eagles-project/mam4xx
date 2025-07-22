#ifndef MAM4XX_DIAGNOSTIC_ARRAYS_HPP
#define MAM4XX_DIAGNOSTIC_ARRAYS_HPP

namespace mam4 {
struct MicrophysDiagnosticArrays {
  using View2D = DeviceType::view_2d<Real>;
  using View1D = DeviceType::view_1d<Real>;

  // each of these views, if non-empty, will be filled by
  // mam4::microphysics::perform_atmospheric_chemistry_and_microphysics
  // dimension number of levels  x  gas_pcnst
  // Gas aerosol exchange from the suffix-indicated process
  View2D gas_aero_exchange_condensation;
  View2D gas_aero_exchange_renaming;
  View2D gas_aero_exchange_nucleation;
  View2D gas_aero_exchange_coagulation;
  // this one is for the cloudwater species
  View2D gas_aero_exchange_renaming_cloud_borne;
  // Tendency due to gas phase (gs_) chemistry [kg/kg/s]
  View2D gas_phase_chemistry_dvmrdt;
  // Tendency due to aqueous (qq_) chemistry [kg/kg/s]
  View2D aqueous_chemistry_dvmrdt;

  // dqdt_so4_aqueous_chemistry[num_modes] So4 flux in kg/m2/s
  // dqdt_h2so4_uptake[num_modes] H2So4 flux in kg/m2/s
  View1D dqdt_so4_aqueous_chemistry;
  View1D dqdt_h2so4_uptake;

  // In-cloud chemistry production/sink of SO4
  View2D aqso4_incloud_mmr_tendency;

  // In-cloud chemistry production/sink of h2so4
  View2D aqh2so4_incloud_mmr_tendency;
};
} // namespace mam4
#endif
