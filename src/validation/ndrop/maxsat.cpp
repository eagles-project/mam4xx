// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/mam4.hpp>

#include <mam4xx/aero_config.hpp>
#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;

void ccncalc(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    // number of vertical points.
    const Real zero = 0;
    const int maxd_aspectype = 14;
    const int ntot_amode = 4;
    const int nvars = 40;

    const int pver = input.get_array("pver")[0];
    const auto state_q_db = input.get_array("state_q");

    const auto tair_db = input.get_array("temp");
    const auto pmid_db = input.get_array("pmid");

    const int top_lev = 6;

    ColumnView state_q[nvars];

    for (int i = 0; i < nvars; ++i) {
      state_q[i] = haero::testing::create_column_view(pver);
    }

    int count = 0;
    // std::vector<std::vector<Real>> state_q_2d;
    for (int i = 0; i < nvars; ++i) {
      // std::vector<Real> temp;
      for (int kk = 0; kk < pver; ++kk) {
        state_q[i](kk) = state_q_db[count];
        // temp.push_back(state_q_db[count]);
        count++;
      }
      // state_q_2d.push_back(temp);
    }

    /*
        real(r8), parameter :: p0 = 1013.25e2_r8    ! reference pressure [Pa]
        real(r8), parameter :: smallest_val = 1.e-39_r8
        real(r8), parameter :: smaller_val = 1.e-20_r8
        real(r8), parameter :: small_val = 1.e-10_r8
        real(r8) pres ! pressure [Pa]
        real(r8) diff0 ! vapor diffusivity [m2/s]
        real(r8) conduct0 ! thermal conductivity [J / (m-s-K)]
        real(r8) es ! saturation vapor pressure [Pa]
        real(r8) qs ! water vapor saturation specific humidity [kg/kg]
        real(r8) dqsdt ! change in qs with temperature  [(kg/kg)/T]
        real(r8) zeta, eta(nmode) ! [unitless]
        real(r8) alpha ! [/m]
        real(r8) gamma ![m3/kg]
        real(r8) beta ! [m2/s]
        real(r8) gthermfac ! thermodynamic function [m2/s]
        real(r8) :: amcube(nmode) ! cube of dry mode radius [m3]
        real(r8) smc(nmode) ! critical supersaturation for number mode radius
       [fraction] real(r8) :: lnsm(nmode) ! ln(critical supersaturation for
       activation) [unitless]

        real(r8) wnuc  ! nucleation w, but = w_in if wdiab == 0 [m/s]
        real(r8) alw ! [/s]
        real(r8) smax ! maximum supersaturation [fraction]
        real(r8) lnsmax ! ln(smax) [unitless]
        real(r8) arg_erf_n,arg_erf_m  ! [unitless]
        real(r8) etafactor1  ! [/ s^(3/2)]
        real(r8) etafactor2(nmode),etafactor2max ! [s^(3/2)]


        !initialize activated aerosols and their fluxes to zero for all the
       modes fn(:)=0._r8 fm(:)=0._r8 fluxn(:)=0._r8 fluxm(:)=0._r8
        flux_fullact=0._r8

        !return if aerosol number is negligible in the accumulation mode
        if(na(1) < smaller_val)return

        !return if vertical velocity is 0 or negative
        if(w_in <= 0._r8)return

        if ( present( smax_prescribed ) ) then
           !return if max supersaturation is 0 or negative
           if (smax_prescribed <= 0.0_r8) return
        endif

        pres=rair*rhoair*tair !pressure
        !Obtain Saturation vapor pressure (es) and saturation specific humidity
       (qs) call qsat(tair, pres, es, qs) !es and qs are the outputs

        dqsdt=latvap/(rh2o*tair*tair)*qs
        alpha=gravit*(latvap/(cpair*rh2o*tair*tair)-1._r8/(rair*tair))
        gamma=(1+latvap/cpair*dqsdt)/(rhoair*qs)
        etafactor2max=1.e10_r8/(alpha*wmaxf)**1.5_r8 !this should make eta big
       if na is very small.

        diff0=0.211e-4_r8*(p0/pres)*(tair/t0)**1.94_r8
        conduct0=(5.69_r8+0.017_r8*(tair-t0))*4.186e2_r8*1.e-5_r8 !convert to
       J/m/s/deg gthermfac=1._r8/(rhoh2o/(diff0*rhoair*qs) &
             +latvap*rhoh2o/(conduct0*tair)*(latvap/(rh2o*tair)-1._r8))
       !gthermfac is same for all modes beta=2._r8*pi*rhoh2o*gthermfac*gamma
        wnuc = w_in
        alw=alpha*wnuc
        etafactor1=alw*sqrt(alw)
        zeta=twothird*sqrt(alw)*aten/sqrt(gthermfac)

        !Here compute smc, eta for all modes for maxsat calculation
        do imode=1,nmode
           if(volume(imode) > smallest_val .and. na(imode) > smallest_val)then
              !number mode radius (m)
              amcube(imode)=(3._r8*volume(imode)/(4._r8*pi*exp45logsig(imode)*na(imode)))
       ! only if variable size dist !Growth coefficent Abdul-Razzak & Ghan 1998
       eqn 16 !should depend on mean radius of mode to account for gas kinetic
       effects !see Fountoukis and Nenes, JGR2005 and Meskhidze et al., JGR2006
              !for approriate size to use for effective diffusivity.

              etafactor2(imode)=1._r8/(na(imode)*beta*sqrt(gthermfac))
              if(hygro(imode) > small_val)then
                 smc(imode)=2._r8*aten*sqrt(aten/(27._r8*hygro(imode)*amcube(imode)))
       ! only if variable size dist else smc(imode)=100._r8 endif else
              smc(imode)=1._r8
              etafactor2(imode)=etafactor2max ! this should make eta big if na
       is very small. endif lnsm(imode)=log(smc(imode)) ! only if variable size
       dist eta(imode)=etafactor1*etafactor2(imode) enddo
        */

    ndrop::maxsat(zeta, eta, nmodes, smc, smax);

    for (int i = 0; i < 6; ++i) {
      ccn[i][kk] = ccn_kk[i];
    }

    } // end kk

    // printf("ccn(%d) 0 %e \n",top_lev, ccn[0][top_lev]);

    for (int i = 0; i < 6; ++i)
    {
    output.set("ccn_" + std::to_string(i + 1), ccn[i]);
    }
});
}
