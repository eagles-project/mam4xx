// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_WET_DEPOSITION_HPP
#define MAM4XX_WET_DEPOSITION_HPP

#include <mam4xx/aero_config.hpp>

namespace mam4 {

namespace wetdep {
/*
        // Need atm for this constant based on number of atmosphere levels
        // See https://github.com/eagles-project/e3sm_mam4_refactor/blob/refactor-maint-2.0/components/eam/src/physics/cam/ppgrid.F90#L32
        const int pver = atm.num_levels();
    // Function should only take and return scalar values
    KOKKOS_INLINE_FUNCTION
    void local_precip_production(cont int ncol, const Real *pdel, const Real source_term, 
                                 const Real sink_term, Real *lprec, const int pver) {
   // do icol=1,ncol
        // icol == num of columns in mesh in sphere
        // we should be only doing one column at this level

        for (int icol = 0; i < ncol; icol++)
        {
            for (int kk = 0; kk < pver;  kk++)
            {
                // TODO find gravit and pass into function
                lprec[icol][kk] = ( pdel[kk] / )
            }
        }
   //    do kk=1,pver
   //       lprec(icol,kk)  = (pdel(icol,kk)/gravit)*(source_term(icol,kk)-sink_term(icol,kk))
   //    enddo
   // enddo
    }
*/
} // namespace wetdep

class WetDeposition {
    public:
        struct Config {
            Config() = default;
            Config(const Config &) = default;
            ~Config() = default;
            Config &operator=(const Config &) = default;
        };

    private:
        Config config_;

    public:
        const char *name() const { return "MAM4 Wet Deposition"; }
        
        void init(const AeroConfig &aero_config,
                  const Config &wed_dep_config = Config()) {
            config_ = wed_dep_config;
        }
};

} // namespace mam4

#endif
