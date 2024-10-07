#ifndef MAM4XX_VERTICAL_INTERPOLATION_HPP
#define MAM4XX_VERTICAL_INTERPOLATION_HPP
#include <haero/math.hpp>
namespace mam4 {

namespace vertical_interpolation {
using View1D = DeviceType::view_1d<Real>;
using ConstView1D = DeviceType::view_1d<const Real>;
using View1DInt = DeviceType::view_1d<int>;

// Port of components/eam/src/chemistry/utils/tracer_data.F90/vert_interp
// The original routine was serial for columns and levs;
// this version uses parallel_for for columns and levels.

KOKKOS_INLINE_FUNCTION
void vert_interp(const ThreadTeam &team, int levsiz, int pver,
                 const View1D &pin, const ConstView1D &pmid,
                 const View1D &datain, const View1D &dataout) {
  const int zero = 0;

  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, pver), [&](int k) {
    // Top level we need to start looking is the top level for the previous k
    // for all column points
    int kupper = zero;
    int kkstart = zero;
    // // Store level indices for interpolation
    for (int kk = kkstart; kk < levsiz - 1; ++kk) {
      if (pin(kk) < pmid(k) && pmid(k) <= pin(kk + 1)) {
        kupper = kk;
      } // end if
    }   // end for

    if (pmid(k) < pin(0)) {
      EKAT_KERNEL_ASSERT_MSG(
          pin(0) != 0.0,
          "Error: Division by zero. The value of pin(0) is zero.\n");
      dataout(k) = datain(0) * pmid(k) / pin(0);
    } else if (pmid(k) > pin(levsiz - 1)) {
      dataout(k) = datain(levsiz - 1);
    } else {
      Real dpu = pmid(k) - pin(kupper);
      Real dpl = pin(kupper + 1) - pmid(k);
      EKAT_KERNEL_ASSERT_MSG(
          dpl + dpu != 0.0,
          "Error: Division by zero. The value of dpl + dpu is zero.\n");
      dataout(k) =
          (datain(kupper) * dpl + datain(kupper + 1) * dpu) / (dpl + dpu);
    } // end if
  }); // end k

} // vert_interp
// rebin is a port from:
// https://github.com/eagles-project/e3sm_mam4_refactor/blob/ee556e13762e41a82cb70a240c54dc1b1e313621/components/eam/src/chemistry/utils/mo_util.F90#L12
KOKKOS_INLINE_FUNCTION
void rebin(const ThreadTeam &team, int nsrc, int ntrg, const ConstView1D &src_x,
           const Real trg_x[], const View1D &src, const View1D &trg) {

  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, ntrg), [&](int i) {
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
      EKAT_KERNEL_ASSERT_MSG(trg_x[i + 1] - trg_x[i] != 0.0,
                             "Error: Division by zero. The value of trg_x[i + "
                             "1] - trg_x[i] is zero.\n");
      trg(i) = y / (trg_x[i + 1] - trg_x[i]);
    } else {
      trg(i) = 0.0;
    }
  });
} // rebin
} // namespace vertical_interpolation

} // end namespace mam4

#endif
