# vcpkg's toolchain, with optimizations in the Debug configuration. The configuration keeps its CRT (/MDd) and _DEBUG, so
# these libraries can still be linked statically with Debug code.
# VCPKG_CXX_FLAGS_DEBUG can't do this: windows.cmake always adds /RTC1, which MSVC refuses to combine with /O2.
list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES _VCPKG_ROOT_DIR) # This file also runs in try_compile projects
include("${_VCPKG_ROOT_DIR}/scripts/toolchains/windows.cmake")

# Projects with a recent policy version (CMP0184) get their /RTC1 from here instead of the flags below
set(CMAKE_MSVC_RUNTIME_CHECKS "")

foreach(flags CMAKE_C_FLAGS_DEBUG CMAKE_CXX_FLAGS_DEBUG)
	string(REPLACE "/Ob0 /Od /RTC1" "/O2 /Ob2" optimized "${${flags}}")
	set(${flags} "${optimized}" CACHE STRING "" FORCE)
endforeach()
