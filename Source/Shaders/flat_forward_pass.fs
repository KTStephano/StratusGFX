STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

smooth in vec2 fsTexCoords;

#include "common.glsl"
#include "alpha_test.glsl"

//uniform vec3 diffuseColor;

flat in int fsDrawID;

//uniform float gamma = 2.2;

out vec3 color;

void main() {
    Material material = materials[materialIndices[fsDrawID]];
    vec4 diffuse = material.diffuseColor;
    if (bitwiseAndBool(material.flags, GPU_DIFFUSE_MAPPED)) {
        diffuse = texture(material.diffuseMap, fsTexCoords);
    }

    runAlphaTest(diffuse.a, 0.25);

    // Apply gamma correction
    //texColor = pow(texColor, vec3(1.0 / gamma));
    color = diffuse.rgb;

    // Small offset to help prevent z fighting in certain cases
    if (diffuse.a < 1.0) {
        gl_FragDepth = gl_FragCoord.z - 0.00001;
    }
    else {
        gl_FragDepth = gl_FragCoord.z;
    }
}