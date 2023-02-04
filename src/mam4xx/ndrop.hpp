subroutine get_aer_num(imode, istart, istop, state_q, cs, vaerosol, qcldbrn1d_num, &!in
           naerosol) !out

  use modal_aero_data, only:numptr_amode

  ! input arguments
  integer,  intent(in) :: imode        ! mode index
  integer,  intent(in) :: istart       ! start column index (1 <= istart <= istop <= pcols)
  integer,  intent(in) :: istop        ! stop column index
  real(r8), intent(in) :: state_q(:,:) ! interstitial aerosol number mixing ratios [#/kg]
  real(r8), intent(in) :: cs(:)        ! air density [kg/m3]
  real(r8), intent(in) :: vaerosol(:)  ! volume conc [m3/m3]
  real(r8), intent(in) :: qcldbrn1d_num(:) ! cloud-borne aerosol number mixing ratios [#/kg]

  !output arguments
  real(r8), intent(out) :: naerosol(:)  ! number conc [#/m3]

  !internal
  integer  :: icol, num_idx

  !convert number mixing ratios to number concentrations
  !Use bulk volume conc found previously to bound value

  num_idx = numptr_amode(imode)
  do icol = istart, istop
     naerosol(icol) = (state_q(icol,num_idx) + qcldbrn1d_num(icol))*cs(icol)
     !adjust number so that dgnumlo < dgnum < dgnumhi
     naerosol(icol) = max(naerosol(icol), vaerosol(icol)*voltonumbhi_amode(imode))
     naerosol(icol) = min(naerosol(icol), vaerosol(icol)*voltonumblo_amode(imode))
  enddo
end subroutine get_aer_num


KOKKOS_INLINE_FUNCTION
void get_aer_num(const Diagnostics &diags,
                                const Prognostics &progs, int mode_idx, int k, Real vaerosol) {
  //out
  Real naerosol = 0.0; // number concentration [#/m3]


  for (int aid = 0; aid < AeroConfig::num_aerosol_ids(); ++aid) {
    const int s = aerosol_index_for_mode(static_cast<ModeIndex>(mode_idx),
                                         static_cast<AeroId>(aid));

    if(s >= 0) {

        Real rho = conversions::density_of_ideal_gas(haero::Atmosphere.temperature[s], haero::Atmosphere.pressure[s]);
        Real vaerosol = haero::AeroSpecies.density;
        naerosol[s] = (progs.q_aero_i[mode_idx][s](k) + progs.q_aero_c[mode_idx][s](k)) * rho;
        //adjust number so that dgnumlo < dgnum < dgnumhi
        naerosol[s] = max(naerosol[s], vaerosol*voltonumbhi_amode[mode_idx]);
        naerosol[s] = min(naerosol[s], vaerosol*voltonumblo_amode[mode_dix]);
    }

  }
}