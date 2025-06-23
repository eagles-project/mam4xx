#ifndef MAM4XX_DIAGNOSTIC_ARRAYS_HPP
#define MAM4XX_DIAGNOSTIC_ARRAYS_HPP

namespace mam4 {
struct MicrophysDiagnosticArrays {
  using View2D = DeviceType::view_2d<Real>;
  // Tendency due to gas phase (gs_) chemistry [kg/kg/s]
  // if non-empty will be filled by
  // mam4::microphysics::perform_atmospheric_chemistry_and_microphysics
  // dimension number of levels  x  gas_pcnst
  View2D gs_dvmrdt;
  // Tendency due to aqueous (qq_) chemistry [kg/kg/s]
  // if non-empty will be filled by
  // mam4::microphysics::perform_atmospheric_chemistry_and_microphysics
  // dimension number of levels  x  gas_pcnst
  View2D aq_dvmrdt;
};
} // namespace mam4
#endif
