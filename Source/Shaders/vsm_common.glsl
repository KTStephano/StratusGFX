STRATUS_GLSL_VERSION

#define VSM_PAGE_RESIDENT_BIT 1
#define VSM_PAGE_DIRTY_BIT 2

// Total is this number squared
#define VSM_MAX_NUM_VIRTUAL_PAGES_XY 32768

// Total is this number squared
#define VSM_MAX_NUM_PHYSICAL_PAGES_XY 128

struct PageResidencyEntry {
    uint frameMarker;
    uint info;
};

// See https://stackoverflow.com/questions/3417183/modulo-of-negative-numbers
ivec2 wrapIndex(ivec2 value, ivec2 maxValue) {
    return ivec2(mod(mod(value, maxValue) + maxValue, maxValue));
}