// validation test implementation of mam4::seq_drydep::setHCoeff
#include <mam4xx/seq_drydep.hpp>
namespace mam4::seq_drydep {
KOKKOS_INLINE_FUNCTION void setHCoeff(Real sfc_temp, Real heff[maxspc]) {
  // Here we populate heff with test data taken from
  // mam_x_validation/mo_drydep/*.yaml (as of Dec 21, 2023). In this dataset,
  // there are 4 ensembles

  static int ensemble = 0;
  if (ensemble == 0) {
    heff[0] = 215141.5904869365;
    heff[1] = 210402677778.856;
    heff[2] = 3334.148860680581;
  } else if (ensemble == 1) {
    heff[0] = 31138.432794480996;
    heff[1] = 42992261164.98709;
    heff[2] = 874.5427866732105;
  } else if (ensemble == 2) {
    heff[0] = 1724166.2082613558;
    heff[1] = 1163173586735.6272;
    heff[2] = 14108.669620617442;
  } else {
    heff[0] = 63920.21390215724;
    heff[1] = 77625658959.18587;
    heff[2] = 1438.7290493476496;
  }
  ++ensemble;
}
} // namespace mam4::seq_drydep
