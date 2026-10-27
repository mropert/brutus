#pragma once

#include <array>
#include <numeric>
#include <renderer/command_buffer.h>
#include <span>
#include <utility>

enum class GpuTiming : uint32_t
{
	FRAME = 0,
	CULLING_SHADOWS,
	DRAW_SHADOWS,
	CULLING_EARLY,
	DRAW_OPAQUES_EARLY,
	DEPTH_PYRAMID,
	CULLING_LATE,
	DRAW_OPAQUES_LATE,
	DRAW_TRANSPARENTS,
	DRAW_GRID,
	TONEMAP,
	DRAW_UI,
	COUNT
};

struct GpuTimings
{
	static constexpr auto TIMINGS_COUNT = std::to_underlying( GpuTiming::COUNT );
	static constexpr auto TIMESTAMPS_COUNT = TIMINGS_COUNT * 2;

	explicit GpuTimings( float gpu_timestamp_period )
		: period( gpu_timestamp_period )
	{
	}

	void add( std::span<const uint64_t, TIMESTAMPS_COUNT> values, int frame_index )
	{
		const int index = frame_index % last_values[ 0 ].size();
		for ( size_t i = 0; i < last_values.size(); ++i )
		{
			last_values[ i ][ index ] = ( values[ i * 2 + 1 ] - values[ i * 2 ] ) * period;
		}
		for ( size_t i = 0; i < last_values.size(); ++i )
		{
			averages[ i ] = std::accumulate( begin( last_values[ i ] ), end( last_values[ i ] ), 0.f ) / last_values[ i ].size();
		}
	}

	std::array<std::array<float, 60>, TIMINGS_COUNT> last_values;
	std::array<float, TIMINGS_COUNT> averages;
	float period;
};

class GpuTimingScope
{
public:
	GpuTimingScope( renderer::CommandBuffer& cmd, renderer::TimestampQuery pool, GpuTiming scope )
		: _cmd( cmd )
		, _pool( pool )
		, _index( std::to_underlying( scope ) )
	{
		_cmd.write_timestamp( _pool, _index * 2 );
	}
	~GpuTimingScope() { end(); }

	GpuTimingScope( const GpuTimingScope& ) = delete;
	GpuTimingScope( GpuTimingScope&& ) = delete;
	GpuTimingScope& operator=( const GpuTimingScope& ) = delete;
	GpuTimingScope& operator=( GpuTimingScope&& ) = delete;

	void end()
	{
		if ( _pool )
		{
			_cmd.write_timestamp( _pool, _index * 2 + 1 );
			_pool = nullptr;
		}
	}

private:
	renderer::CommandBuffer& _cmd;
	renderer::TimestampQuery _pool;
	uint32_t _index;
};