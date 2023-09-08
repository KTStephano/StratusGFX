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

#define VSM_PAGE_FRAME_MARKER_MASK 0xFFFFFFF0
#define VSM_PAGE_FRAME_UPDATE_MASK 0x0000000F

// Total is this number squared
#define VSM_MAX_NUM_VIRTUAL_PAGES_XY 32760
#define VSM_MAX_VALUE_VIRTUAL_PAGE_XY 16380

// Total is this number squared
#define VSM_MAX_NUM_PHYSICAL_PAGES_XY 128

#define VSM_MAX_NUM_TEXELS_PER_PAGE_XY 128
#define VSM_DOUBLE_NUM_TEXELS_PER_PAGE_XY 256

// 128 * 128
#define VSM_MAX_NUM_TEXELS_PER_PAGE 16384
#define VSM_HALF_NUM_TEXELS_PER_PAGE 8192
#define VSM_FOURTH_NUM_TEXELS_PER_PAGE 4096

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

vec2 wrapUVCoords(in vec2 uv) {
    return fract(uv);
}

// See https://developer.download.nvidia.com/books/HTML/gpugems/gpugems_ch11.html
float sampleShadowTexture(sampler2DArrayShadow shadow, vec4 coords, float depth, vec2 offset, float bias) {
    coords.w = depth - bias;
    coords.xy += offset;
    return texture(shadow, coords);
}

float sampleShadowTextureSparse(sampler2DArrayShadow shadow, vec4 coords, float depth, vec2 offset, float bias) {
    coords.w = depth - bias;
    coords.xy += offset;
    //coords.xy = wrapIndex(coords.xy, vec2(1.0 + PREVENT_DIV_BY_ZERO));
    float result;
    int status = sparseTextureARB(shadow, coords, result);
    return (sparseTexelsResidentARB(status) == false) ? 0.0 : result;
    //return result;
}

// float sampleShadowTextureSparse(in sampler2DArray shadow, in vec4 coords, in float depth, in vec2 offset, in float bias) {
//     ivec2 size = textureSize(shadow, 0).xy;

//     float offsetDepth = depth - bias;

//     vec4 modifiedCoords = coords;
//     // modifiedCoords.w = depth - bias;
//     modifiedCoords.xy += offset;

//     // See https://gamedev.stackexchange.com/questions/101953/low-quality-bilinear-sampling-in-webgl-opengl-directx
//     float fractU = fract(modifiedCoords.x);
//     float fractV = fract(modifiedCoords.y);

//     //modifiedCoords.xy = wrapIndex(modifiedCoords.xy, vec2(1.0 + PREVENT_DIV_BY_ZERO));
//     modifiedCoords.xy = modifiedCoords.xy * vec2(size - ivec2(1));
//     //modifiedCoords.xy = wrapIndex(modifiedCoords.xy * vec2(size - ivec2(1)), vec2(size));

//     float numSamplesPassed = 0.0;
//     vec4 result = vec4(0.0);
//     int status = 0;

//     status = sparseTexelFetchARB(
//         shadow, 
//         ivec3(wrapIndex(modifiedCoords.xy, vec2(size)), modifiedCoords.z),
//         0, 
//         result
//     );
//     float sampledDepth1 = sparseTexelsResidentARB(status) == false ? 0.0 : result.r;

//     status = sparseTexelFetchARB(
//         shadow, 
//         ivec3(wrapIndex(modifiedCoords.xy + vec2(1, 0), vec2(size)), modifiedCoords.z),
//         0, 
//         result
//     );
//     float sampledDepth2 = sparseTexelsResidentARB(status) == false ? 0.0 : result.r;

//     status = sparseTexelFetchARB(
//         shadow, 
//         ivec3(wrapIndex(modifiedCoords.xy + vec2(0, 1), vec2(size)), modifiedCoords.z),
//         0, 
//         result
//     );
//     float sampledDepth3 = sparseTexelsResidentARB(status) == false ? 0.0 : result.r;

//     status = sparseTexelFetchARB(
//         shadow, 
//         ivec3(wrapIndex(modifiedCoords.xy + vec2(1, 1), vec2(size)), modifiedCoords.z),
//         0, 
//         result
//     );
//     float sampledDepth4 = sparseTexelsResidentARB(status) == false ? 0.0 : result.r;

//     // status = sparseTexelFetchARB(
//     //     shadow, 
//     //     ivec3(wrapIndex(modifiedCoords.xy + vec2(1, -1), vec2(size)), modifiedCoords.z),
//     //     0, 
//     //     result
//     // );
//     // float sampledDepth5 = sparseTexelsResidentARB(status) == false ? 0.0 : result.r;

//     numSamplesPassed += (offsetDepth <= sampledDepth1) ? 0.5 : 0.0;
//     numSamplesPassed += (offsetDepth <= sampledDepth2) ? 1.0 : 0.0;
//     numSamplesPassed += (offsetDepth <= sampledDepth3) ? 1.0 : 0.0;
//     numSamplesPassed += (offsetDepth <= sampledDepth4) ? 1.0 : 0.0;
//     // //numSamplesPassed += (offsetDepth <= sampledDepth5) ? 1.0 : 0.0;

//     return numSamplesPassed / 9.0;
// }

// float sampleShadowTextureSparse(in sampler2DArray shadow, in vec4 coords, in float depth, in vec2 offset, in float bias) {

//     ivec2 size = textureSize(shadow, 0).xy;
//     float offsetDepth = depth - bias;

//     vec4 modifiedCoords = coords;
//     // modifiedCoords.w = depth - bias;
//     modifiedCoords.xy += offset;

//     //modifiedCoords.xy = wrapIndex(modifiedCoords.xy, vec2(1.0 + PREVENT_DIV_BY_ZERO));
//     modifiedCoords.xy = modifiedCoords.xy * vec2(size - ivec2(1));
//     //modifiedCoords.xy = wrapIndex(modifiedCoords.xy * vec2(size - ivec2(1)), vec2(size));

//     float numSamplesPassed = 0.0;
//     vec4 result = vec4(0.0);
//     int status = 0;

//     status = sparseTexelFetchARB(
//         shadow, 
//         ivec3(wrapIndex(modifiedCoords.xy, vec2(size)), modifiedCoords.z),
//         0, 
//         result
//     );
    
//     if (sparseTexelsResidentARB(status) == false) return 0.0;

//     return offsetDepth <= result.r ? 1.0 : 0.0;
// }

// float sampleShadowTexture(in sampler2DArray shadow, in vec4 coords, in float depth, in vec2 offset, in float bias) {
//     return sampleShadowTextureSparse(shadow, coords, depth, offset, bias);
// }

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

uint packFrameCountWithUpdateCount(in uint frameCount, in uint updateCount) {
    return (frameCount << 4) | (updateCount & VSM_PAGE_FRAME_UPDATE_MASK);
}

void unpackFrameCountAndUpdateCount(in uint data, out uint frameCount, out uint updateCount) {
    frameCount = data >> 4;
    updateCount = data & VSM_PAGE_FRAME_UPDATE_MASK;
}

vec2 roundIndex(in vec2 index) {
    return index;
    //return index - vec2(1.0);
    //return ceil(index) - vec2(1.0);
    //return round(index) - vec2(1.0);
    //return floor(index);
}

// vec2 convertVirtualCoordsToPhysicalCoordsNoRound(
//     in ivec2 virtualCoords, 
//     in ivec2 maxVirtualIndex
// ) {
    
//     // We need to convert our virtual texel to a physical texel
//     vec2 virtualTexCoords = vec2(virtualCoords + ivec2(1)) / vec2(maxVirtualIndex + ivec2(1));

//     // Set up NDC using -1, 1 tex coords and -1 for the z coord
//     vec4 ndc = vec4(virtualTexCoords * 2.0 - 1.0, 0.0, 1.0);

//     // Subtract off the translation since the orientation should be
//     // the same for all vsm clip maps - just translation changes
//     vec2 ndcOrigin = ndc.xy - vsmClipMap0ProjectionView[3].xy;
    
//     // Convert from [-1, 1] to [0, 1]
//     vec2 physicalTexCoords = ndcOrigin * 0.5 + vec2(0.5);

//     return wrapIndex(physicalTexCoords * vec2(maxVirtualIndex), vec2(maxVirtualIndex + ivec2(1)));
// }
vec2 convertVirtualCoordsToPhysicalCoordsNoRound(
    in vec2 virtualCoords, 
    in vec2 maxVirtualIndex,
    in int cascadeIndex
) {
    
    // We need to convert our virtual texel to a physical texel
    //vec2 virtualTexCoords = vec2(virtualCoords) / vec2(maxVirtualIndex);

    // Set up NDC using -1, 1 tex coords and -1 for the z coord
    //vec4 ndc = vec4(virtualTexCoords * 2.0 - 1.0, 0.0, 1.0);
    vec4 ndc = vec4(vec2(2.0 * virtualCoords) / vec2(maxVirtualIndex + 1) - 1.0, 0.0, 1.0);

    // Subtract off the translation since the orientation should be
    // the same for all vsm clip maps - just translation changes
    vec2 ndcOrigin = ndc.xy - vsmConvertClip0ToClipN(vsmClipMap0ProjectionView[3].xy, cascadeIndex);
    //ndcOrigin = vsmConvertClip0ToClipN(ndcOrigin, cascadeIndex);
    
    // Convert from [-1, 1] to [0, 1]
    vec2 physicalTexCoords = wrapUVCoords(ndcOrigin * 0.5 + vec2(0.5));

    return physicalTexCoords * vec2(maxVirtualIndex);
    //return wrapIndex(physicalTexCoords * vec2(maxVirtualIndex), vec2(maxVirtualIndex + vec2(1)));
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

// vec2 convertPhysicalCoordsToVirtualCoordsNoRound(
//     in ivec2 physicalCoords, 
//     in ivec2 maxPhysicalIndex
// ) {
    
//     // We need to convert our virtual texel to a physical texel
//     vec2 physicalTexCoords = vec2(physicalCoords + ivec2(1)) / vec2(maxPhysicalIndex + ivec2(1));

//     // Set up NDC using -1, 1 tex coords and -1 for the z coord
//     vec4 ndc = vec4(physicalTexCoords * 2.0 - 1.0, 0.0, 1.0);

//     // Add back the translation component to convert physical ndc to relative virtual ndc
//     vec2 ndcRelative = ndc.xy + vsmClipMap0ProjectionView[3].xy;
    
//     // Convert from [-1, 1] to [0, 1]
//     vec2 virtualTexCoords = ndcRelative * 0.5 + vec2(0.5);

//     return wrapIndex(virtualTexCoords * vec2(maxPhysicalIndex), vec2(maxPhysicalIndex + ivec2(1)));
// }
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

    return virtualTexCoords * vec2(maxPhysicalIndex);
    //return wrapIndex(virtualTexCoords * vec2(maxPhysicalIndex), vec2(maxPhysicalIndex + vec2(1)));
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