STRATUS_GLSL_VERSION

#define VSM_PAGE_DIRTY_BIT 1
#define VSM_PAGE_DIRTY_MASK 0x00000001
#define VSM_PAGE_ID_MASK 0xFFFFFFFE

// Total is this number squared
#define VSM_MAX_NUM_VIRTUAL_PAGES_XY 32768

// Total is this number squared
#define VSM_MAX_NUM_PHYSICAL_PAGES_XY 128

struct PageResidencyEntry {
    uint frameMarker;
    uint info;
};

// See https://stackoverflow.com/questions/3417183/modulo-of-negative-numbers
ivec2 wrapIndex(in ivec2 value, in ivec2 maxValue) {
    return ivec2(mod(mod(value, maxValue) + maxValue, maxValue));
}

uint computePageId(in ivec2 page) {
    return uint(page.x + page.y * VSM_MAX_NUM_VIRTUAL_PAGES_XY);
}

uint packPageIdWithDirtyBit(in uint pageId, in uint bit) {
    return (pageId << 1) | (bit & VSM_PAGE_DIRTY_MASK);
}

void unpackPageIdAndDirtyBit(in uint data, out uint pageId, out uint bit) {
    pageId = data >> 1;
    bit = data & VSM_PAGE_DIRTY_MASK;
}