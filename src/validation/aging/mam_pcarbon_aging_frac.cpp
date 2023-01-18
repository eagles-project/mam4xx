#include <mam4xx/calcsize.hpp>
#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;

void mam_pcarbon_aging_frac(Ensemble *ensemble){

  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {

  int nlev = 1;
  mam4::Prognostics progs(nlev);

  const auto dgn_a_f = input.get_array("dgn_a");
  const auto qaer_cur_f = input.get_array("qaer_cur");
  const auto qaer_del_cond_f = input.get_array("qaer_del_cond");
  const auto qaer_del_coag_in_f = input.get_array("qaer_del_coag_in");

  const int num_modes = AeroConfig::num_modes(); 
  const int num_aero = AeroConfig::num_aerosol_ids();

  Real qaer_cur_c[num_aero][num_modes]; 
  Real qaer_del_cond_c[num_aero][num_modes];
  Real qaer_del_coag_in_c[num_aero][num_modes];

  for (int a=0; a < num_aero; ++a){
    for (int m=0; m< num_modes; ++m ){

    }
  }

  
    

 });
}