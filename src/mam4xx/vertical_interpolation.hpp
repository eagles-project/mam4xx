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
void vert_interp(const int icol, int levsiz, int pver, const View2D &pin,
                 const ConstView2D &pmid, const View2D &datain,
                 const View2D &dataout,
                 // work array
                 const View1DInt &kupper
                 ) {
  const int zero = 0;
  kupper(icol) = zero;
  for (int k = 0; k < pver; ++k) {
    // Top level we need to start looking is the top level for the previous k
    // for all column points
    int kkstart = levsiz - 1;
    kkstart = haero::min(kkstart, kupper(icol));
    // // Store level indices for interpolation
    for (int kk = kkstart; kk < levsiz - 1; ++kk) {
        if (pin(icol, kk) < pmid(icol, k) && pmid(icol, k) <= pin(icol, kk + 1)) {
          kupper(icol) = kk;
        } // end if
      }   // end for

      if (pmid(icol, k) < pin(icol, 0)) {
        dataout(icol, k) = datain(icol, 0) * pmid(icol, k) / pin(icol, 0);
      } else if (pmid(icol, k) > pin(icol, levsiz - 1)) {
        dataout(icol, k) = datain(icol, levsiz - 1);
      } else {
        Real dpu = pmid(icol, k) - pin(icol, kupper(icol));
        Real dpl = pin(icol, kupper(icol) + 1) - pmid(icol, k);
        dataout(icol, k) =
            (datain(icol, kupper(icol)) * dpl + datain(icol, kupper(icol) + 1) * dpu) /
            (dpl + dpu);
      } // end if
    }   // end k

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
