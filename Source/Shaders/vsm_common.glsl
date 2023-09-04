STRATUS_GLSL_VERSION

#include "bindings.glsl"
#include "common.glsl"

#define VSM_LOWER_MASK 0x0000FFFF
#define VSM_UPPER_MASK 0xFFFF0000

#define VSM_PAGE_DIRTY_BIT 1
#define VSM_PAGE_DIRTY_MASK VSM_LOWER_MASK
#define VSM_PAGE_ID_MASK VSM_UPPER_MASK

// Total is this number squared
#define VSM_MAX_NUM_VIRTUAL_PAGES_XY 32768

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
        return original * type(1.0 / float(BITMASK_POW2(clipIndex))); \
    }

VSM_CONVERT_CLIP0_TO_CLIP_N(vec2)
VSM_CONVERT_CLIP0_TO_CLIP_N(vec3)
VSM_CONVERT_CLIP0_TO_CLIP_N(vec4)

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

    return result - vsmClipMap0ProjectionView[3].xyz;
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

uint computePageId(in ivec2 page) {
    return uint(VSM_MAX_NUM_VIRTUAL_PAGES_XY + page.x + page.y * VSM_MAX_NUM_PHYSICAL_PAGES_XY);
}

uint packPageIdWithDirtyBit(in uint pageId, in uint bit) {
    return (pageId << 16) | (bit & VSM_PAGE_DIRTY_MASK);
}

void unpackPageIdAndDirtyBit(in uint data, out uint pageId, out uint bit) {
    pageId = (data >> 16) & VSM_LOWER_MASK;
    bit = data & VSM_PAGE_DIRTY_MASK;
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
    in ivec2 virtualCoords, 
    in ivec2 maxVirtualIndex,
    in int cascadeIndex
) {
    
    // We need to convert our virtual texel to a physical texel
    vec2 virtualTexCoords = vec2(virtualCoords) / vec2(maxVirtualIndex);

    // Set up NDC using -1, 1 tex coords and -1 for the z coord
    vec4 ndc = vec4(virtualTexCoords * 2.0 - 1.0, 0.0, 1.0);

    // Subtract off the translation since the orientation should be
    // the same for all vsm clip maps - just translation changes
    vec2 ndcOrigin = ndc.xy - vsmClipMap0ProjectionView[3].xy;
    ndcOrigin = vsmConvertClip0ToClipN(ndcOrigin, cascadeIndex);
    
    // Convert from [-1, 1] to [0, 1]
    vec2 physicalTexCoords = ndcOrigin * 0.5 + vec2(0.5);

    return wrapIndex(physicalTexCoords * vec2(maxVirtualIndex), vec2(maxVirtualIndex + ivec2(1)));
}

vec2 convertVirtualCoordsToPhysicalCoords(
    in ivec2 virtualCoords, 
    in ivec2 maxVirtualIndex,
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
    in ivec2 physicalCoords, 
    in ivec2 maxPhysicalIndex,
    in int cascadeIndex
) {
    
    // We need to convert our virtual texel to a physical texel
    vec2 physicalTexCoords = vec2(physicalCoords) / vec2(maxPhysicalIndex);

    // Set up NDC using -1, 1 tex coords and -1 for the z coord
    vec4 ndc = vec4(physicalTexCoords * 2.0 - 1.0, 0.0, 1.0);

    // Add back the translation component to convert physical ndc to relative virtual ndc
    vec2 ndcRelative = ndc.xy + vsmClipMap0ProjectionView[3].xy;
    ndcRelative = vsmConvertClip0ToClipN(ndcRelative, cascadeIndex);
    
    // Convert from [-1, 1] to [0, 1]
    vec2 virtualTexCoords = ndcRelative * 0.5 + vec2(0.5);

    return wrapIndex(virtualTexCoords * vec2(maxPhysicalIndex), vec2(maxPhysicalIndex + ivec2(1)));
}

vec2 convertPhysicalCoordsToVirtualCoords(
    in ivec2 physicalCoords, 
    in ivec2 maxPhysicalIndex,
    in int cascadeIndex
) {
    
    vec2 wrapped = convertPhysicalCoordsToVirtualCoordsNoRound(
        physicalCoords,
        maxPhysicalIndex,
        cascadeIndex
    );

    return roundIndex(wrapped);
}