STRATUS_GLSL_VERSION

#include "bindings.glsl"
#include "common.glsl"
#include "aabb.glsl"

#define VSM_LOWER_MASK 0x00000003
#define VSM_UPPER_MASK 0xFFFFFFFC

#define VSM_PAGE_DIRTY_BIT 1
#define VSM_PAGE_CLEARED_BIT 2
#define VSM_PAGE_RENDERED_BIT 3
#define VSM_PAGE_DIRTY_MASK VSM_LOWER_MASK
#define VSM_PAGE_ID_MASK VSM_UPPER_MASK

#define VSM_PAGE_FRAME_MARKER_MASK 0xF0000000
#define VSM_PAGE_RESIDENCY_STATUS_MASK 0x0000000F

#define VSM_PAGE_X_OFFSET_MASK 0x0FF00000
#define VSM_PAGE_Y_OFFSET_MASK 0x000FF000
#define VSM_PAGE_MEM_POOL_OFFSET_MASK 0x00000FF0

// Total is this number squared
#define VSM_MAX_NUM_VIRTUAL_PAGES_XY 32760
#define VSM_MAX_VALUE_VIRTUAL_PAGE_XY 16380

// Total is this number squared
#define VSM_MAX_NUM_PHYSICAL_PAGES_XY 128

#define VSM_MAX_NUM_TEXELS_PER_PAGE_XY 128
//#define VSM_DOUBLE_NUM_TEXELS_PER_PAGE_XY 256

// 128 * 128
//#define VSM_MAX_NUM_TEXELS_PER_PAGE 16384
// #define VSM_HALF_NUM_TEXELS_PER_PAGE 8192
// #define VSM_FOURTH_NUM_TEXELS_PER_PAGE 4096

// #define ATOMIC_REDUCE_TEXEL_COUNT(info) {                         \
//     uint pageId;                                                  \
//     uint count;                                                   \
//     unpackPageIdAndDirtyBit(info, pageId, count);                 \
//     uint original = packPageIdWithDirtyBit(pageId, count);        \
//     while (count > 0) {                                           \
//         --count;                                                  \
//         uint updated = packPageIdWithDirtyBit(pageId, count);     \
//         uint prev = atomicCompSwap(info, original, updated);      \
//         if (prev == original) break;                              \
//         unpackPageIdAndDirtyBit(prev, pageId, count);             \
//     }                                                             \
// }

struct PageResidencyEntry {
    uint frameMarker;
    uint info;
};

struct ClipMapBoundingBox {
    int minPageX;
    int minPageY;
    int maxPageX;
    int maxPageY;
};

layout (std430, binding = VSM_CURR_FRAME_RESIDENCY_TABLE_BINDING) coherent buffer vsmPageTableBuffer {
    PageResidencyEntry currFramePageResidencyTable[];
};

// For first clip map - rest are derived from this
uniform mat4 vsmClipMap0ProjectionView;
uniform uint vsmNumCascades;

#define VSM_CONVERT_CLIP0_TO_CLIP_N(type)                             \
    type vsmConvertClip0ToClipN(in type original, in int clipIndex) { \
        type result = original;                                       \
        result.xy *= vec2(1.0 / float(BITMASK_POW2(clipIndex)));      \
        return result;                                                \
    }

VSM_CONVERT_CLIP0_TO_CLIP_N(vec2)
VSM_CONVERT_CLIP0_TO_CLIP_N(vec3)
VSM_CONVERT_CLIP0_TO_CLIP_N(vec4)

float vsmConvertRelativeDepthToOriginDepth(in float depth) {
    return ((2.0 * depth - 1.0) - vsmClipMap0ProjectionView[3].z) * 0.5 + 0.5;
}

vec3 vsmCalculateClipValueFromWorldPos(in mat4 viewProj, in vec3 worldPos, in int clipMapIndex) {
    vec4 result = viewProj * vec4(worldPos, 1.0);

    // Accounts for the fact that each clip map covers double the range of the
    // previous
    result.xy = vsmConvertClip0ToClipN(result.xy, clipMapIndex);

    return result.xyz;
}

// The difference between this and Origin function is that this will return a value
// relative to current clip pos, whereas Origin assumes clip pos = vec3(0.0)
vec3 vsmCalculateRelativeClipValueFromWorldPos(in vec3 worldPos, in int clipMapIndex) {
    return vsmCalculateClipValueFromWorldPos(vsmClipMap0ProjectionView, worldPos, clipMapIndex);
}

// Returns 3 values on the range [-1, 1]
vec3 vsmCalculateOriginClipValueFromWorldPos(in vec3 worldPos, in int clipMapIndex) {
    vec3 result = vsmCalculateRelativeClipValueFromWorldPos(worldPos, clipMapIndex);

    return result - vsmConvertClip0ToClipN(vsmClipMap0ProjectionView[3].xyz, clipMapIndex);
}

int vsmCalculateCascadeIndexFromWorldPos(in vec3 worldPos) {
    vec2 ndc = vsmCalculateRelativeClipValueFromWorldPos(worldPos, 0).xy;

    // This finds positive whole integer solutions to the equation:
    //      ndc * (1 / 2^i) = 1
    // where i is the cascade index
    vec2 cascadeIndex = vec2(
        ceil(log2(max(ceil(abs(ndc.x)), 1.0))),
        ceil(log2(max(ceil(abs(ndc.y)), 1.0)))
    );

    return int(max(cascadeIndex.x, cascadeIndex.y));
}

// See https://stackoverflow.com/questions/3417183/modulo-of-negative-numbers
ivec2 wrapIndex(in ivec2 value, in ivec2 maxValue) {
    return ivec2(mod(mod(value, maxValue) + maxValue, maxValue));
}

vec2 wrapIndex(in vec2 value, in vec2 maxValue) {
    return vec2(mod(mod(value, maxValue) + maxValue, maxValue));
}

vec2 wrapUVCoords(in vec2 uvs) {
    return fract(uvs);
}

uint computePageId(in ivec2 page) {
    ivec2 offsetPage = page + ivec2(VSM_MAX_VALUE_VIRTUAL_PAGE_XY);
    return uint(offsetPage.x + offsetPage.y * VSM_MAX_NUM_VIRTUAL_PAGES_XY);
}

uint packPageIdWithDirtyBit(in uint pageId, in uint bit) {
    return (pageId << 2) | (bit & VSM_PAGE_DIRTY_MASK);
}

void unpackPageIdAndDirtyBit(in uint data, out uint pageId, out uint bit) {
    pageId = data >> 2;
    bit = data & VSM_PAGE_DIRTY_MASK;
}

uint packPageMarkerData(in uint frameCount, in uint pageX, in uint pageY, in uint memPool, in uint residencyStatus) {
    return (frameCount << 28) | 
           ((pageX << 20) & VSM_PAGE_X_OFFSET_MASK) | 
           ((pageY << 12) & VSM_PAGE_Y_OFFSET_MASK) | 
           ((memPool << 4) & VSM_PAGE_MEM_POOL_OFFSET_MASK) |
           (residencyStatus & VSM_PAGE_RESIDENCY_STATUS_MASK);
}

void unpackPageMarkerData(in uint data, out uint frameCount, out uint pageX, out uint pageY, out uint memPool, out uint residencyStatus) {
    frameCount = (data & VSM_PAGE_FRAME_MARKER_MASK) >> 28;
    pageX = (data & VSM_PAGE_X_OFFSET_MASK) >> 20;
    pageY = (data & VSM_PAGE_Y_OFFSET_MASK) >> 12;
    memPool = (data & VSM_PAGE_MEM_POOL_OFFSET_MASK) >> 4;
    residencyStatus = data & VSM_PAGE_RESIDENCY_STATUS_MASK;
}

// See https://developer.download.nvidia.com/books/HTML/gpugems/gpugems_ch11.html
float sampleShadowTexture(sampler2DArrayShadow shadow, vec4 coords, float depth, vec2 offset, float bias) {
    coords.w = depth - bias;
    coords.xy += offset;
    return texture(shadow, coords);
}

// float sampleShadowTextureSparse(sampler2DArrayShadow shadow, vec4 coords, float depth, vec2 offset, float bias) {
//     coords.w = depth - bias;
//     coords.xy += offset;
//     //coords.xy = wrapIndex(coords.xy, vec2(1.0 + PREVENT_DIV_BY_ZERO));
//     float result;
//     int status = sparseTextureARB(shadow, coords, result);
//     return (sparseTexelsResidentARB(status) == false) ? 0.0 : result;
//     //return result;
// }

vec3 vsmConvertVirtualUVToPhysicalPixelCoords(in vec2 uv, in vec2 resolution, in uint vsmNumPagesXY, in int cascadeIndex) {
    vec2 wrappedUvs = wrapUVCoords(uv);
    vec2 virtualPixelCoords = wrappedUvs * resolution;

    //vec2 offsetWithinPage = wrapIndex(virtualPixelCoords, vec2(VSM_MAX_NUM_TEXELS_PER_PAGE_XY));
    vec2 offsetWithinPage = mod(virtualPixelCoords, vec2(VSM_MAX_NUM_TEXELS_PER_PAGE_XY));

    //vec2 physicalPageUv = 2.0 * wrappedUvs - vec2(1.0); //((2.0 * virtualPixelCoords) / resolution - 1.0) * 0.5 + 0.5;
    //ivec2 physicalPageCoords = ivec2(floor(physicalPageUv * vec2(vsmNumPagesXY)));
    ivec2 physicalPageCoords = ivec2(floor(wrappedUvs * vec2(vsmNumPagesXY)));
    // ivec2 physicalPageCoords = ivec2(floor(virtualPixelCoords / vec2(VSM_MAX_NUM_TEXELS_PER_PAGE_XY)));
    uint physicalPageIndex = uint(physicalPageCoords.x + physicalPageCoords.y * vsmNumPagesXY + uint(cascadeIndex) * vsmNumPagesXY * vsmNumPagesXY);

    uint unused1;
    uint physicalOffsetX;
    uint physicalOffsetY;
    uint memPool;
    uint unused3;
    unpackPageMarkerData(
        currFramePageResidencyTable[physicalPageIndex].frameMarker,
        unused1,
        physicalOffsetX,
        physicalOffsetY,
        memPool,
        unused3
    );

    return vec3(
        (vec2(physicalOffsetX, physicalOffsetY) * vec2(VSM_MAX_NUM_TEXELS_PER_PAGE_XY)) + offsetWithinPage,
        float(memPool)
    );
}

float sampleShadowTextureSparse(sampler2DArray shadow, in vec4 coords, float depth, vec2 offset, float bias) {
    float offsetDepth = depth - bias;

    vec4 modified = coords;
    modified.w = depth - bias;
    modified.xy += offset;
    //coords.xy = wrapIndex(coords.xy, vec2(1.0 + PREVENT_DIV_BY_ZERO));

    uint vsmNumPagesXY = uint(textureSize(shadow, 0).x / VSM_MAX_NUM_TEXELS_PER_PAGE_XY);
    vec3 physicalCoords = vsmConvertVirtualUVToPhysicalPixelCoords(
        modified.xy, 
        vec2(textureSize(shadow, 0).xy),
        vsmNumPagesXY,
        int(coords.z)
    );

    vec3 physicalUvs = vec3(
        physicalCoords.xy / vec2(textureSize(shadow, 0).xy),
        physicalCoords.z
    );

    vec4 texel;
    int status = sparseTexelFetchARB(shadow, ivec3(floor(physicalCoords)), 0, texel);

    if (sparseTexelsResidentARB(status) == false) return 0.0;

    return offsetDepth < texel.r ? 1.0 : 0.0;

    // float result;
    // int status = sparseTextureARB(shadow, vec4(physicalUvs, modified.w), result);
    // return (sparseTexelsResidentARB(status) == false) ? 0.0 : result;
}

vec2 roundIndex(in vec2 index) {
    return index;
    //return index - vec2(1.0);
    //return ceil(index) - vec2(1.0);
    //return round(index) - vec2(1.0);
    //return floor(index);
}

vec2 convertLocalCoordsToVirtualUvCoords(
    in vec2 localCoords,
    in vec2 size,
    in int cascadeIndex
) {
    vec2 ndc = (2.0 * localCoords) / size - 1.0;
    vec2 ndcOrigin = ndc.xy - vsmConvertClip0ToClipN(vsmClipMap0ProjectionView[3].xy, cascadeIndex);
    return ndcOrigin * 0.5 + vec2(0.5);
}

vec2 convertVirtualCoordsToPhysicalCoordsNoRound(
    in vec2 virtualCoords, 
    in vec2 maxVirtualIndex,
    in int cascadeIndex
) {
    vec2 physicalTexCoords = wrapUVCoords(convertLocalCoordsToVirtualUvCoords(
        virtualCoords,
        maxVirtualIndex + vec2(1.0),
        cascadeIndex
    ));

    return physicalTexCoords * vec2(maxVirtualIndex + vec2(1));
    // return wrapIndex(
    //     physicalTexCoords * (maxVirtualIndex + vec2(1)), 
    //     maxVirtualIndex + vec2(1)
    // );
}

vec2 convertVirtualCoordsToPhysicalCoords(
    in vec2 virtualCoords, 
    in vec2 maxVirtualIndex,
    in int cascadeIndex
) {
    
    vec2 wrapped = convertVirtualCoordsToPhysicalCoordsNoRound(
        virtualCoords,
        maxVirtualIndex,
        cascadeIndex
    );

    return roundIndex(wrapped);
}

vec2 convertPhysicalCoordsToVirtualCoordsNoRound(
    in vec2 physicalCoords, 
    in vec2 maxPhysicalIndex,
    in int cascadeIndex
) {
    
    // We need to convert our virtual texel to a physical texel
    //vec2 physicalTexCoords = vec2(physicalCoords) / vec2(maxPhysicalIndex);

    // Set up NDC using -1, 1 tex coords and -1 for the z coord
    //vec4 ndc = vec4(physicalTexCoords * 2.0 - 1.0, 0.0, 1.0);
    vec4 ndc = vec4(vec2(2.0 * physicalCoords) / vec2(maxPhysicalIndex + 1) - 1.0, 0.0, 1.0);

    // Add back the translation component to convert physical ndc to relative virtual ndc
    vec2 ndcRelative = ndc.xy + vsmConvertClip0ToClipN(vsmClipMap0ProjectionView[3].xy, cascadeIndex);
    //ndcRelative = vsmConvertClip0ToClipN(ndcRelative, cascadeIndex);
    
    // Convert from [-1, 1] to [0, 1]
    vec2 virtualTexCoords = wrapUVCoords(ndcRelative * 0.5 + vec2(0.5));

    return virtualTexCoords * vec2(maxPhysicalIndex + vec2(1));
    // return wrapIndex(
    //     virtualTexCoords * (maxPhysicalIndex + vec2(1)), 
    //     maxPhysicalIndex + vec2(1)
    // );
}

vec2 convertPhysicalCoordsToVirtualCoords(
    in vec2 physicalCoords, 
    in vec2 maxPhysicalIndex,
    in int cascadeIndex
) {
    
    vec2 wrapped = convertPhysicalCoordsToVirtualCoordsNoRound(
        physicalCoords,
        maxPhysicalIndex,
        cascadeIndex
    );

    return roundIndex(wrapped);
}

void vsmComputeCornersWithTransform(in AABB aabb, in mat4 transform, inout vec4 corners[8]) {
    vec4 vmin = aabb.vmin;
    vec4 vmax = aabb.vmax;

    corners[0] = transform * vec4(vmin.x, vmin.y, vmin.z, 1.0);
    corners[1] = transform * vec4(vmin.x, vmax.y, vmin.z, 1.0);
    corners[2] = transform * vec4(vmin.x, vmin.y, vmax.z, 1.0);
    corners[3] = transform * vec4(vmin.x, vmax.y, vmax.z, 1.0);
    corners[4] = transform * vec4(vmax.x, vmin.y, vmin.z, 1.0);
    corners[5] = transform * vec4(vmax.x, vmax.y, vmin.z, 1.0);
    corners[6] = transform * vec4(vmax.x, vmin.y, vmax.z, 1.0);
    corners[7] = transform * vec4(vmax.x, vmax.y, vmax.z, 1.0);
}

AABB vsmTransformAabbAsNDCCoords(in AABB aabb, in mat4 transform, in vec4 corners[8], inout int cascadeIndex) {
    vsmComputeCornersWithTransform(aabb, transform, corners);

    vec3 vmin3 = vsmConvertClip0ToClipN(corners[0].xyz, cascadeIndex);
    vec3 vmax3 = vsmConvertClip0ToClipN(corners[0].xyz, cascadeIndex);

    for (int i = 1; i < 8; ++i) {
        vmin3 = min(vmin3, vsmConvertClip0ToClipN(corners[i].xyz, cascadeIndex));
        vmax3 = max(vmax3, vsmConvertClip0ToClipN(corners[i].xyz, cascadeIndex));
    }

    AABB result;
    result.vmin = vec4(vmin3, 1.0);
    result.vmax = vec4(vmax3, 1.0); 

    return result;
}