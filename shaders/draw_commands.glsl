// Draw commands written by mesh_cull.comp (counts and slots) then mesh_commands.comp (commands). Commands are sorted into
// one list per material bucket and LOD: the CPU only knows where each bucket's commands start (its worst case), the
// bucket's LOD lists are packed in that range once their sizes are known

const uint DRAW_BUCKETS_COUNT = DRAW_LISTS_COUNT / MESH_LOD_COUNT;

// Mesh shader path: one indirect draw per list
struct TaskCommandsHeader
{
	uint group_count_x; // VkDrawMeshTasksIndirectCommandEXT, the list's commands count
	uint group_count_y;
	uint group_count_z;
	uint draw_count;
	uint first_command; // Index of the list's first command in commands
};
layout( buffer_reference, std430 ) buffer TaskCommandsBuffer
{
	TaskCommandsHeader headers[ DRAW_LISTS_COUNT ];
	MeshTaskCommand commands[];
};

// Vertex path: one indirect multi draw per bucket, its lists in LOD order
layout( buffer_reference, std430 ) buffer MeshDrawCommandsBuffer
{
	uint list_counts[ DRAW_LISTS_COUNT ];
	uint bucket_counts[ DRAW_BUCKETS_COUNT ];
	MeshDrawCommand commands[];
};

// Where culling put each draw, indexed by draw id
const uint DRAW_NOT_LISTED = 0xFFFFFFFF;
struct DrawSlot
{
	uint list; // DRAW_NOT_LISTED when culled, the drawn early flag is bit 8
	uint first_command; // Relative to the list
};
const uint DRAW_SLOT_LIST_MASK = 0xFF;
const uint DRAW_SLOT_DRAWN_EARLY = 0x100;
layout( buffer_reference, std430 ) buffer DrawSlotsBuffer
{
	DrawSlot slots[];
};
