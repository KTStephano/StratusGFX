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

// See "Implementing a material system" in 3D Graphics Rendering Cookbook
// This *only* uses basic punch through transparency and is not a full transparency solution
void runAlphaTest(float alpha, float alphaThreshold) {
    if (alphaThreshold == 0.0) return;

    mat4 thresholdMatrix = mat4(
        1.0/17.0, 9.0/17.0, 3.0/17.0, 11.0/17.0,
        13.0/17.0, 5.0/17.0, 15.0/17.0, 7.0/17.0,
        4.0/17.0, 12.0/17.0, 2.0/17.0, 10.0/17.0,
        16.0/17.0, 8.0/17.0, 14.0/17.0, 6.0/17.0
    );

    int x = int(mod(gl_FragCoord.x, 4.0));
    int y = int(mod(gl_FragCoord.y, 4.0));
    alpha = clamp(alpha - 0.5 * thresholdMatrix[x][y], 0.0, 1.0);
    if (alpha < alphaThreshold) discard;
}

void main() {
    Material material = materials[materialIndices[fsDrawID]];
    vec4 diffuse = material.diffuseColor;
    if (bitwiseAndBool(material.flags, GPU_DIFFUSE_MAPPED)) {
        diffuse = texture(material.diffuseMap, fsTexCoords);
    }

    runAlphaTest(diffuse.a, 0.5);

    // Apply gamma correction
    //texColor = pow(texColor, vec3(1.0 / gamma));
    color = diffuse.rgb;
}