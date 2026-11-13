#pragma once

#include <renderer/texture.h>
#include <span>

// Both expected size of header and texture data start offset in file
inline constexpr size_t DDS_HEADER_SIZE = 148;
// Minimum alignment for the start of the header
inline constexpr size_t DDS_HEADER_ALIGN = alignof( uint32_t );

// sRGB is how the texture is used: an explicit UNORM or SRGB format must match it, a TYPELESS one takes it.
// header_bytes should be the first HEADER_SIZE bytes of the file
// file_size should be the total file size including the header
// Throws renderer::Error on failure
renderer::Texture::Desc parse_dds( std::span<const std::byte, DDS_HEADER_SIZE> header_bytes, size_t file_size, bool sRGB );

// Convenience overload for files already buffered in memory
renderer::Texture::Desc parse_dds( std::span<const std::byte> bytes, bool sRGB );
