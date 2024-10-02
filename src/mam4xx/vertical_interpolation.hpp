#ifndef MAM4XX_VERTICAL_INTERPOLATION_HPP
#define MAM4XX_VERTICAL_INTERPOLATION_HPP
#include <haero/math.hpp>
namespace mam4 {

namespace vertical_interpolation {
using View2D = DeviceType::view_2d<Real>;
using CosntView2D = DeviceType::view_2d<const Real>;
using View1DInt = DeviceType::view_1d<int>;

// Direct port of components/eam/src/chemistry/utils/tracer_data.F90/vert_interp
// FIXME: I need to convert for loops to Kokkos loops.
KOKKOS_INLINE_FUNCTION
void vert_interp(int ncol, int levsiz, int pver, const View2D &pin,
                 const CosntView2D &pmid, const View2D &datain,
                 const View2D &dataout,
                 // work array
                 View1DInt &kupper) {
  const int zero = 0;
  // Initialize index array
  for (int i = 0; i < ncol; ++i) {
    kupper(i) = zero;
  } // ncol

  for (int k = 0; k < pver; ++k) {
    // Top level we need to start looking is the top level for the previous k
    // for all column points
    int kkstart = levsiz - 1;
    for (int i = 0; i < ncol; ++i) {
      kkstart = haero::min(kkstart, kupper(i));
    }

    // Store level indices for interpolation
    for (int kk = kkstart; kk < levsiz - 1; ++kk) {
      for (int i = 0; i < ncol; ++i) {
        if (pin(i, kk) < pmid(i, k) && pmid(i, k) <= pin(i, kk + 1)) {
          kupper(i) = kk;
        } // end if
      }   // end for
    }     // end kk
    // Interpolate or extrapolate...
    for (int i = 0; i < ncol; ++i) {
      if (pmid(i, k) < pin(i, 0)) {
        dataout(i, k) = datain(i, 0) * pmid(i, k) / pin(i, 0);
      } else if (pmid(i, k) > pin(i, levsiz - 1)) {
        dataout(i, k) = datain(i, levsiz - 1);
      } else {
        Real dpu = pmid(i, k) - pin(i, kupper(i));
        Real dpl = pin(i, kupper(i) + 1) - pmid(i, k);
        dataout(i, k) =
            (datain(i, kupper(i)) * dpl + datain(i, kupper(i) + 1) * dpu) /
            (dpl + dpu);
      } // end if
    }   // end col
  }     // end k

} // vert_interp

} // namespace vertical_interpolation
} // end namespace mam4

#endif
