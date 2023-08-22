STRATUS_GLSL_VERSION

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

#define IMAGE_ATOMIC_MIN_FLOAT_SPARSE(image, coords, data) {                    \
    uvec4 texel;                                                                \
    int status = sparseImageLoadARB(image, coords, texel);                      \
    uint prevCompare = 0;                                                       \
    uint compare = texel.r;                                                     \
    float mem = uintBitsToFloat(compare);                                       \
    uint converted = floatBitsToUint(data);                                     \
    if (sparseTexelsResidentARB(status)) {                                      \
        while (prevCompare != compare && data < mem) {                          \
            prevCompare =  compare;                                             \
            compare = imageAtomicCompSwap(image, coords, compare, converted);   \
            mem = uintBitsToFloat(compare);                                     \
        }                                                                       \
    }                                                                           \
}