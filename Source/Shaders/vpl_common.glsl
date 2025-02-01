STRATUS_GLSL_VERSION

#include "pbr.glsl"

// These needs to match what is in the renderer backend!
// TODO: Find a better way to sync these values with renderer
#define MAX_TOTAL_VPLS_BEFORE_CULLING (MAX_TOTAL_SHADOW_MAPS) //(10000)
#define MAX_TOTAL_VPLS_PER_FRAME (MAX_TOTAL_SHADOW_MAPS)
#define MAX_VPLS_PER_TILE (12)

// Max probes per bucket
#define MAX_VPLS_PER_BUCKET (MAX_TOTAL_VPLS_PER_FRAME)
// Total buckets
#define MAX_VPL_BUCKETS_PER_DIM (1)
#define MAX_VPL_BUCKETS (MAX_VPL_BUCKETS_PER_DIM*MAX_VPL_BUCKETS_PER_DIM*MAX_VPL_BUCKETS_PER_DIM)
#define MAX_VPL_BUCKET_INDEX (MAX_VPL_BUCKETS)
// Each bucket occupies this value cubed in world space
#define WORLD_SPACE_PER_VPL_BUCKET (1024)

// Needs to use uint for its memory backing since GLSL atomics only work on int and uint
// See for reference:
//  https://codereview.stackexchange.com/questions/258883/glsl-atomic-float-add-for-architectures-with-out-atomic-float-add
//  https://registry.khronos.org/OpenGL-Refpages/gl4/html/atomicCompSwap.xhtml
//  https://registry.khronos.org/OpenGL-Refpages/gl4/html/atomicAdd.xhtml
//  https://registry.khronos.org/OpenGL-Refpages/gl4/html/floatBitsToInt.xhtml
//
// Returns the original contents of mem before the add
//
// See https://community.khronos.org/t/atomiccompswap/69213 for why it needs to be a #define
//
// Type of mem: uint
// Type of data: float
// Type of old: uint
#define ATOMIC_ADD_FLOAT(mem, data, old) {                              \
    uint expected = mem;                                                \
    uint added    = floatBitsToUint(uintBitsToFloat(mem) + data);       \
    uint returned = atomicCompSwap(mem, expected, added);               \
    while (expected != returned) {                                      \
        expected = returned;                                            \
        added    = floatBitsToUint(uintBitsToFloat(expected) + data);   \
        returned = atomicCompSwap(mem, expected, added);                \
    }                                                                   \
    old = returned;                                                     \
}

// Needs to match up with definition in StratusGpuCommon
struct VplStage1PerTileOutputs {
    vec4 averageLocalPosition;
    vec4 averageLocalNormal;
};

struct VplStage2PerTileOutputs {
    int numVisible;
    int indices[MAX_VPLS_PER_TILE];
};

struct VplData {
    vec4 position;
    float intensityScale;
    float padding_[3];
};

// Takes uvs coords and wraps them to [0, 1] range
vec3 wrapUVSCoords(in vec3 uvs) {
    return vec3(fract(uvs.x), fract(uvs.y), fract(uvs.z));
}

// Inputs should be in world space
ivec3 computeBaseBucketCoords(in vec3 position) {
    // Converts a position first to bucket index, then to [-1, 1] range with out of bounds allowed
    vec3 normalized = (position / float(WORLD_SPACE_PER_VPL_BUCKET)) / float(MAX_VPL_BUCKETS_PER_DIM);
    // Converts normalized coords to [0, 1] range with coordinate wrapping
    vec3 uvs = wrapUVSCoords(normalized * 0.5 + vec3(0.5));
    return ivec3(floor(uvs*vec3(MAX_VPL_BUCKETS_PER_DIM)));
    //return ivec3(0,0,0);
}

bool baseBucketCoordsWithinRange(in ivec3 bucketCoords) {
    return bucketCoords.x >= 0 && bucketCoords.x < MAX_VPL_BUCKETS_PER_DIM &&
        bucketCoords.y >= 0 && bucketCoords.y < MAX_VPL_BUCKETS_PER_DIM &&
        bucketCoords.z >= 0 && bucketCoords.z < MAX_VPL_BUCKETS_PER_DIM;
}

// Takes input bucket coords and clamps any values outside of [0, MAX_VPL_BUCKETS_PER_DIM) to be
// on the edge
ivec3 clampBaseBucketCoords(in ivec3 bucketCoords) {
    return ivec3(clamp(bucketCoords.x, 0, MAX_VPL_BUCKETS_PER_DIM-1),
                 clamp(bucketCoords.y, 0, MAX_VPL_BUCKETS_PER_DIM-1),
                 clamp(bucketCoords.z, 0, MAX_VPL_BUCKETS_PER_DIM-1));
}

int computeBaseBucketIndex(in ivec3 bucketCoords) {
    return bucketCoords.x + 
           bucketCoords.y * MAX_VPL_BUCKETS_PER_DIM +
           bucketCoords.z * MAX_VPL_BUCKETS_PER_DIM * MAX_VPL_BUCKETS_PER_DIM;
}

int computeOffsetBucketIndex(in int index) {
    return index * MAX_VPLS_PER_BUCKET;
}