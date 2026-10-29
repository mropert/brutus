#pragma once

#include <renderer/texture.h>
#include <span>

class DDSFile
{
public:
	// Throws renderer::Error on failure
	explicit DDSFile( std::span<const std::byte> bytes );

	const renderer::Texture::Desc& get_desc() const { return _desc; }
	std::span<const std::byte> get_data() const;

private:
	std::span<const std::byte> _bytes;
	renderer::Texture::Desc _desc;
};
