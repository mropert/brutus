// Draw commands written by mesh_cull.comp (counts and slots) then mesh_commands.comp (commands). Commands are split by
// bucket, the material type, then sorted into bins by a sort key (today the LOD): the CPU only knows where each bucket's
// commands start (its worst case), the bucket's bins are packed in that range once their sizes are known

const uint DRAW_BINS_COUNT = DRAW_BUCKETS_COUNT * DRAW_BINS_PER_BUCKET;

// Mesh shader path: one indirect draw per bin
struct TaskCommandsHeader
{
	uint group_count_x; // VkDrawMeshTasksIndirectCommandEXT, the bin's commands count
	uint group_count_y;
	uint group_count_z;
	uint draw_count;
	uint first_command; // Index of the bin's first command in commands
};
layout( buffer_reference, std430 ) buffer TaskCommandsBuffer
{
	TaskCommandsHeader headers[ DRAW_BINS_COUNT ];
	MeshTaskCommand commands[];
};

// Vertex path: one indirect multi draw per bucket, its bins in order
layout( buffer_reference, std430 ) buffer MeshDrawCommandsBuffer
{
	uint bin_counts[ DRAW_BINS_COUNT ];
	uint bucket_counts[ DRAW_BUCKETS_COUNT ];
	MeshDrawCommand commands[];
};

// Where culling put each draw, indexed by draw id
const uint DRAW_NOT_BINNED = 0xFFFFFFFF;
struct DrawSlot
{
	uint bin; // DRAW_NOT_BINNED when culled, the drawn early flag is bit 8
	uint offset_in_bin; // Where the draw's commands start in its bin
};
const uint DRAW_SLOT_BIN_MASK = 0xFF;
const uint DRAW_SLOT_DRAWN_EARLY = 0x100;
layout( buffer_reference, std430 ) buffer DrawSlotsBuffer
{
	DrawSlot slots[];
};
