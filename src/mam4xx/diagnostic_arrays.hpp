#ifndef MAM4XX_DIAGNOSTIC_ARRAYS_HPP
#define MAM4XX_DIAGNOSTIC_ARRAYS_HPP

namespace mam4 {
struct MicrophysDiagnosticArrays {
  using View2D = DeviceType::view_2d<Real>;
  using View1D = DeviceType::view_1d<Real>;
  // Tendency due to gas phase (gs_) chemistry [mole/mole/s]
  // if non-empty will be filled by
  // mam4::microphysics::perform_atmospheric_chemistry_and_microphysics
  // dimension number of levels  x  gas_pcnst
  View2D gas_phase_chemistry_dvmrdt;
  // Tendency due to aqueous (qq_) chemistry [kg/kg/s]
  // if non-empty will be filled by
  // mam4::microphysics::perform_atmospheric_chemistry_and_microphysics
  // dimension number of levels  x  gas_pcnst
  View2D aqueous_chemistry_dvmrdt;

  // deposition flux of so4 [mole/mole/s]
  View1D aqso4_column_integrated_flux;

  // deposition flux of h2so4 [mole/mole/s]
  View1D aqh2so4_column_integrated_flux;
};
} // namespace mam4
#endif
