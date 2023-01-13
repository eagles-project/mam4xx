#ifndef MAM4XX_AGING_HPP
#define MAM4XX_AGING_HPP
#include <mam4xx/aero_config.hpp>
#include <mam4xx/conversions.hpp>
#include <mam4xx/merikanto2007.hpp>
#include <mam4xx/vehkamaki2002.hpp>
#include <mam4xx/wang2008.hpp>

#include <haero/atmosphere.hpp>
#include <haero/math.hpp>


namespace mam4{

class Aging{
public:


 struct Config{

    Config() {}; 

    Config(const Config &) = default;
    ~Config() = default;
    Config &operator=(const Config &) = default;
 };

private:
  Config config_;


public:
  // name -- unique name of the process implemented by this class
  const char *name() const { return "MAM4 aging"; }

   // In E3SM this is read in from an input file
  static constexpr Real n_so4_monolayers_pcage  =  8.0;   
  static constexpr Real dr_so4_monolayers_pcage  =  n_so4_monolayers_pcage * 4.76e-10;


  void init(const AeroConfig &aero_config,
            const Config &aging_config = Config()) {

        config_ = aging_config;

        
    };


  // validate -- validates the given atmospheric state and prognostics against
  // assumptions made by this implementation, returning true if the states are
  // valid, false if not
  KOKKOS_INLINE_FUNCTION
  bool validate(const AeroConfig &config, const ThreadTeam &team,
                const Atmosphere &atm, const Prognostics &progs) const;


  // compute_tendencies -- computes tendencies and updates diagnostics
  // NOTE: that both diags and tends are const below--this means their views
  // NOTE: are fixed, but the data in those views is allowed to vary.
  KOKKOS_INLINE_FUNCTION
  void compute_tendencies(const AeroConfig &config, const ThreadTeam &team,
                          Real t, Real dt, const Atmosphere &atm,
                          const Prognostics &progs, const Diagnostics &diags,
                          const Tendencies &tends) const;


};

namespace aging{


KOKKOS_INLINE_FUNCTION
void mam_pcarbon_aging_frac(
     Real dgn_a[AeroConfig::num_modes()],
     Real qaer_cur[AeroConfig::num_aerosol_ids()][AeroConfig::num_modes()],
     Real qaer_del_cond[AeroConfig::num_aerosol_ids()][AeroConfig::num_modes()],
     Real qaer_del_coag_in[AeroConfig::num_aerosol_ids()][AeroConfig::num_modes()],
     Real &xferfrac_pcage, 
     Real &frac_cond,
     Real &frac_coag
     )                                                                
{       

  const int ipair = 1;
  static constexpr int nsrc = static_cast<int>(ModeIndex::PrimaryCarbon); 
  const int iaer_so4 = aerosol_index_for_mode(ModeIndex::PrimaryCarbon, AeroId::SO4);
  const int iaer_soa = aerosol_index_for_mode(ModeIndex::PrimaryCarbon, AeroId::SOA); 
  const int iaer_bc = aerosol_index_for_mode(ModeIndex::PrimaryCarbon, AeroId::BC); 
  const int iaer_pom = aerosol_index_for_mode(ModeIndex::PrimaryCarbon, AeroId::POM); 
  const int iaer_mom = aerosol_index_for_mode(ModeIndex::PrimaryCarbon, AeroId::MOM); 


  // Compute the aerosol volume per mole 
  const Real so4_vol = aero_species(iaer_so4).molecular_weight/aero_species(iaer_so4).density;
  const Real soa_vol =  aero_species(iaer_soa).molecular_weight/aero_species(iaer_soa).density;
  const Real bc_vol =  aero_species(iaer_bc).molecular_weight/aero_species(iaer_bc).density;
  const Real pom_vol =  aero_species(iaer_pom).molecular_weight/aero_species(iaer_pom).density;
  const Real mom_vol =  aero_species(iaer_mom).molecular_weight/aero_species(iaer_mom).density;

  const Real fac_m2v_eqvhyg_aer = soa_vol * aero_species(iaer_soa).hygroscopicity; 


  const Real vol_shell = qaer_cur[iaer_so4][nsrc] * so4_vol + qaer_cur[iaer_soa][nsrc] * fac_m2v_eqvhyg_aer;
  const Real qaer_del_cond_tmp =  haero::max( qaer_del_cond[iaer_so4][nsrc]*so4_vol + qaer_del_cond[iaer_soa][nsrc]*fac_m2v_eqvhyg_aer, std::numeric_limits<float>::epsilon()) ;
  const Real qaer_del_coag_tmp = qaer_del_coag_in[iaer_so4][ipair]*so4_vol + qaer_del_coag_in[iaer_soa][ipair]*fac_m2v_eqvhyg_aer;
  
  frac_cond = 1.0 - qaer_del_cond_tmp/(qaer_del_cond_tmp + haero::max( qaer_del_coag_tmp, 0.0));
  frac_coag = 1.0 - frac_cond;

  const int core_modes[3] = {iaer_bc, iaer_pom, iaer_mom};
  const Real core_volumes[3] = {bc_vol, pom_vol, mom_vol};
  Real vol_core = 0.0; 
  for (int mi =0; mi<3; ++mi ){
    const int m = core_modes[mi];
    vol_core += qaer_cur[m][nsrc] * core_volumes[m];
  }
 
  const Real fac_volsfc = haero::exp(2.5 * haero::log(haero::pow(mam4::modes(nsrc).mean_std_dev,2.0)));
  const Real xferfrac_max = 1.0 - 10.0 * std::numeric_limits<float>::epsilon();  //  1-eps

  Real xferfrac_tmp1 = vol_shell*dgn_a[nsrc]*fac_volsfc; 
  Real xferfrac_tmp2 = haero::max( 6.0*Aging::dr_so4_monolayers_pcage*vol_core, 0.0);

  if (xferfrac_tmp1 >= xferfrac_tmp2){
    xferfrac_pcage = xferfrac_max;
  } else{
    xferfrac_pcage =  haero::min( xferfrac_tmp1/xferfrac_tmp2, xferfrac_max ); 
  }

  (void) xferfrac_max ;
  (void)  xferfrac_tmp1; 
  (void)  xferfrac_tmp2; 
  (void) fac_volsfc; 
  (void) vol_shell;
  (void)  frac_cond;
  (void) vol_core;


//(void) vol_shell;

// const Real qaer_del_cond_tmp = qaer_del_cond[iaer_so4][nsrc]*mass_2_vol[iaer_so4] + qaer_del_cond[iaer_soa][nsrc]*fac_m2v_eqvhyg_aer[iaer_soa];

// const Real qaer_del_coag_tmp = qaer_del_coag_in[iaer_so4][ipair]*mass_2_vol[iaer_so4] + qaer_del_coag_in[iaer_soa][ipair]*fac_m2v_eqvhyg_aer[iaer_soa];


}


KOKKOS_INLINE_FUNCTION
void mam_pcarbon_aging_1subarea(
                               Real dgn_a[AeroConfig::num_modes()],
                               Real qnum_cur[AeroConfig::num_modes()],
                               Real qnum_del_cond[AeroConfig::num_modes()],
                               Real qnum_del_coag[AeroConfig::num_aerosol_ids()][AeroConfig::num_modes()],
                               Real qaer_cur[AeroConfig::num_aerosol_ids()][AeroConfig::num_modes()],
                               Real qaer_del_cond[AeroConfig::num_aerosol_ids()][AeroConfig::num_modes()],
                               Real qaer_del_coag_in[AeroConfig::num_aerosol_ids()][AeroConfig::num_modes()]
){

  Real xferfrac_pcage; 
  Real frac_cond;
  Real frac_coag; 

  //static constexpr int ndest = static_cast<int>(ModeIndex::Accumulation);  // Source is accumulation mode
  //static constexpr int nsrc = static_cast<int>(ModeIndex::PrimaryCarbon);  // Destination in primary carbon mode

  mam_pcarbon_aging_frac(dgn_a, qaer_cur, qaer_del_cond, qaer_del_coag_in, xferfrac_pcage, frac_cond, frac_coag);

}

           





KOKKOS_INLINE_FUNCTION
void transfer_aged_pcarbon_to_accum(int nsrc, int ndest, Real xferfrac_pcage, 
Real frac_cond, Real frac_coag, Real *q_cur, Real *q_del_cond, Real *q_del_coag){

const Real q_tmp = q_cur[nsrc]*xferfrac_pcage; 

q_cur[nsrc] -=  q_tmp;
q_cur[ndest] +=  q_tmp;

q_del_cond[nsrc] -= q_tmp*frac_cond; 
q_del_cond[ndest] += q_tmp*frac_cond;

q_del_coag[nsrc] -= q_tmp*frac_coag;
q_del_coag[nsrc] += q_tmp*frac_coag;

}

KOKKOS_INLINE_FUNCTION
void transfer_cond_coag_mass_to_accum(int nsrc, int ndest,
Real *qaer_cur, Real *qaer_del_cond, Real *qaer_del_coag){


qaer_cur[ndest] += qaer_cur[nsrc];
qaer_del_cond[ndest] += qaer_del_cond[nsrc];
qaer_del_coag[ndest] += qaer_del_cond[nsrc];

qaer_cur[nsrc] = 0.0;
qaer_del_cond[nsrc] = 0.0;
qaer_del_coag[nsrc] = 0.0;

}

}
}
#endif