# Stock x64-windows, except for the slowest dependencies at load: they're optimized in Debug too, see optimized_debug.cmake
set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE dynamic)
set(VCPKG_PROVIDED_FORTRAN ON)

if(PORT MATCHES "^(libpng|zlib|shaderc|glslang|spirv-tools)$")
	set(VCPKG_CHAINLOAD_TOOLCHAIN_FILE "${CMAKE_CURRENT_LIST_DIR}/optimized_debug.cmake")
	set(VCPKG_LOAD_VCVARS_ENV ON) # Not implied with a chainloaded toolchain
endif()
