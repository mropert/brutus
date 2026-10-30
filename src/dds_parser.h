#pragma once

#include <renderer/texture.h>
#include <span>

class DDSFile
{
public:
	// sRGB is how the texture is used: an explicit UNORM or SRGB format must match it, a TYPELESS one takes it.
	// Throws renderer::Error on failure
	DDSFile( std::span<const std::byte> bytes, bool sRGB );

	const renderer::Texture::Desc& get_desc() const { return _desc; }
	std::span<const std::byte> get_data() const;

private:
	std::span<const std::byte> _bytes;
	renderer::Texture::Desc _desc;
};
