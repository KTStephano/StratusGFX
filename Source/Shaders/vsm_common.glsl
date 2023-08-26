STRATUS_GLSL_VERSION

#define VSM_PAGE_DIRTY_BIT 1
#define VSM_PAGE_DIRTY_MASK 0x00000001
#define VSM_PAGE_ID_MASK 0xFFFFFFFE

// Total is this number squared
#define VSM_MAX_NUM_VIRTUAL_PAGES_XY 32768

// Total is this number squared
#define VSM_MAX_NUM_PHYSICAL_PAGES_XY 128

#define VSM_MAX_NUM_TEXELS_PER_PAGE_XY 128

struct PageResidencyEntry {
    uint frameMarker;
    uint info;
};

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
    return (pageId << 1) | (bit & VSM_PAGE_DIRTY_MASK);
}

void unpackPageIdAndDirtyBit(in uint data, out uint pageId, out uint bit) {
    pageId = data >> 1;
    bit = data & VSM_PAGE_DIRTY_MASK;
}

vec2 roundIndex(in vec2 index) {
    return ceil(index) - vec2(1.0);
}

vec2 convertVirtualCoordsToPhysicalCoords(
    in ivec2 virtualPixelCoords, 
    in ivec2 maxVirtualIndex, 
    in mat4 invProjectionView, 
    in mat4 vsmProjectionView
) {
    
    // We need to convert our virtual texel to a physical texel
    vec2 virtualTexCoords = vec2(virtualPixelCoords + ivec2(1)) / vec2(maxVirtualIndex + ivec2(1));
    // Set up NDC using -1, 1 tex coords and -1 for the z coord
    vec4 ndc = vec4(virtualTexCoords * 2.0 - 1.0, 0.0, 1.0);
    // Convert to world space
    vec4 worldPosition = invProjectionView * ndc;
    // Perspective divide
    worldPosition.xyz /= worldPosition.w;

    vec4 physicalTexCoords = vsmProjectionView * vec4(worldPosition.xyz, 1.0);
    // Perspective divide
    physicalTexCoords.xy = physicalTexCoords.xy / physicalTexCoords.w;
    // Convert from range [-1, 1] to [0, 1]
    physicalTexCoords.xy = physicalTexCoords.xy * 0.5 + vec2(0.5);

    vec2 wrapped = wrapIndex(physicalTexCoords.xy * vec2(maxVirtualIndex), vec2(maxVirtualIndex + ivec2(1)));
    return roundIndex(wrapped);
}