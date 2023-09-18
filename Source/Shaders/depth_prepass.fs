STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require
#extension GL_ARB_sparse_texture2 : require

//layout (early_fragment_tests) in;

#include "common.glsl"
#include "alpha_test.glsl"

smooth in vec2 fsTexCoords;
flat in int fsDrawID;

flat in int fsDiffuseMapped;

void main() {
    Material material = materials[materialIndices[fsDrawID]];
    vec4 baseColor = bool(fsDiffuseMapped) ? texture(material.diffuseMap, fsTexCoords) : decodeMaterialData(material.diffuseColor);
    runAlphaTest(baseColor.a);
}