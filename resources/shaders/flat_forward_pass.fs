STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

smooth in vec2 fsTexCoords;

#include "common.glsl"

//uniform vec3 diffuseColor;
// Material information
layout (std430, binding = 11) readonly buffer SSBO1 {
    Material materials[];
};

layout (std430, binding = 12) readonly buffer SSBO2 {
    uint materialIndices[];
};

flat in int fsDrawID;

//uniform float gamma = 2.2;

out vec3 color;

void main() {
    Material material = materials[materialIndices[fsDrawID]];
    vec3 diffuse = material.diffuseColor.xyz;
    if (bitwiseAndBool(material.flags, GPU_DIFFUSE_MAPPED)) {
        diffuse = texture(material.diffuseMap, fsTexCoords).xyz;
    }
    // Apply gamma correction
    //texColor = pow(texColor, vec3(1.0 / gamma));
    color = diffuse;
}