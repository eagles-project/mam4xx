include(GNUInstallDirs)

# Most of mam4xx is implemented in C++ headers, so we must
# install them for a client.
install(FILES
        aero_config.hpp
        aero_modes.hpp
        calcsize.hpp
        conversions.hpp
        gasaerexch.hpp
        gasaerexch_soaexch.hpp
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
        wv_sat_methods.hpp
        DESTINATION include/mam4xx)

add_library(mam4xx aero_modes.cpp)
install(TARGETS mam4xx DESTINATION ${CMAKE_INSTALL_LIBDIR}/mam4xx)
