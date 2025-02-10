STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

//layout (early_fragment_tests) in;

#include "common.glsl"
#include "alpha_test.glsl"

smooth in vec2 fsTexCoords;
flat in int fsDrawID;
flat in int fsDiffuseMapped;

void main() {
    Material material = materials[materialIndices[fsDrawID]];
    uint flags = material.flags;

    vec4 baseColor = bool(fsDiffuseMapped) ? texture(material.diffuseMap, fsTexCoords) : FLOAT4_TO_VEC4(material.diffuseColor);
    runAlphaTest(baseColor.a);

    // Small offset to help prevent z fighting in certain cases
    //gl_FragDepth = baseColor.a < 1.0 ? gl_FragCoord.z - ALPHA_DEPTH_OFFSET : gl_FragCoord.z;
}