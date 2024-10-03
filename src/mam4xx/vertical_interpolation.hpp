#ifndef MAM4XX_VERTICAL_INTERPOLATION_HPP
#define MAM4XX_VERTICAL_INTERPOLATION_HPP
#include <haero/math.hpp>
namespace mam4 {

namespace vertical_interpolation {
using View1D = DeviceType::view_1d<Real>;
using View2D = DeviceType::view_2d<Real>;
using ConstView2D = DeviceType::view_2d<const Real>;
using ConstView1D = DeviceType::view_1d<const Real>;
using View1DInt = DeviceType::view_1d<int>;

// Direct port of components/eam/src/chemistry/utils/tracer_data.F90/vert_interp
// FIXME: I need to convert for loops to Kokkos loops.
KOKKOS_INLINE_FUNCTION
void vert_interp(int ncol, int levsiz, int pver, const View2D &pin,
                 const ConstView2D &pmid, const View2D &datain,
                 const View2D &dataout,
                 // work array
                 const View1DInt &kupper) {
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
// rebin is a port from:
// https://github.com/eagles-project/e3sm_mam4_refactor/blob/ee556e13762e41a82cb70a240c54dc1b1e313621/components/eam/src/chemistry/utils/mo_util.F90#L12
KOKKOS_INLINE_FUNCTION
void rebin(int nsrc, int ntrg, const ConstView1D &src_x, const Real trg_x[],
           const View1D &src, const View1D &trg) {
  for (int i = 0; i < ntrg; ++i) {
    Real tl = trg_x[i];
    if (tl < src_x(nsrc)) {
      int sil = 0;
      for (; sil <= nsrc; ++sil) {
        if (tl <= src_x(sil)) {
          break;
        }
      }
      Real tu = trg_x[i + 1];
      int siu = 0;
      for (; siu <= nsrc; ++siu) {
        if (tu <= src_x(siu)) {
          break;
        }
      }
      Real y = 0.0;
      sil = haero::max(sil, 1);
      siu = haero::min(siu, nsrc);
      for (int si = sil; si <= siu; ++si) {
        int si1 = si - 1;
        Real sl = haero::max(tl, src_x(si1));
        Real su = haero::min(tu, src_x(si));
        y += (su - sl) * src(si1);
      }
      trg(i) = y / (trg_x[i + 1] - trg_x[i]);
    } else {
      trg(i) = 0.0;
    }
  }
} // rebin
} // namespace vertical_interpolation

} // end namespace mam4

#endif
