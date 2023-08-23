include(GNUInstallDirs)

# Generate aero_config.hpp, which contains build-time parameters.
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/aero_config.hpp.in
  ${CMAKE_CURRENT_BINARY_DIR}/aero_config.hpp
  @ONLY
)

# Most of mam4xx is implemented in C++ headers, so we must
# install them for a client.
install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/aero_config.hpp
        aero_model.hpp
        aero_modes.hpp
        calcsize.hpp
        conversions.hpp
        convproc.hpp
        gasaerexch.hpp
        gasaerexch_soaexch.hpp
        gas_chem.hpp
        gas_chem_mechanism.hpp
        kerminen2002.hpp
        kohler.hpp
        mam4.hpp
        mam4_types.hpp
        merikanto2007.hpp
        nucleate_ice.hpp
        nucleation.hpp
        aging.hpp
        coagulation.hpp
        rename.hpp
        utils.hpp
        ndrop.hpp
        vehkamaki2002.hpp
        wang2008.hpp
        hetfrz.hpp
        wv_sat_methods.hpp
        drydep.hpp
        water_uptake.hpp
        wet_dep.hpp
        ndrop.hpp
        mo_photo.hpp
        lin_strat_chem.hpp
        mo_chm_diags.hpp
        mo_setext.hpp
        mo_setsox.hpp
        DESTINATION include/mam4xx)

add_library(mam4xx aero_modes.cpp)
install(TARGETS mam4xx DESTINATION ${CMAKE_INSTALL_LIBDIR})
