# Everything in mam4xx so far is implemented in C++ headers, so we must
# install them for a client.
install(FILES
        aero_config.hpp
        aero_modes.hpp
        calcsize.hpp
        conversions.hpp
        gasaerexch.hpp
        kerminen2002.hpp
        kohler.hpp
        mam4.hpp
        mam4_types.hpp
        merikanto2007.hpp
        nucleation.hpp
        vehkamaki2002.hpp
        wang2008.hpp
        DESTINATION include/mam4xx)
