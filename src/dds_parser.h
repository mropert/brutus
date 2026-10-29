#pragma once

#include <renderer/texture.h>
#include <span>

class dds_file
{
public:
	explicit dds_file( std::span<const std::byte> bytes );

	// Calling any accessor when this returns false is UB
	explicit constexpr operator bool() const { return !_bytes.empty(); }

	renderer::Texture::Desc get_desc() const;
	std::span<const std::byte> get_data() const;

private:
	std::span<const std::byte> _bytes;
};
