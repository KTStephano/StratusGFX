STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

smooth in vec2 fsTexCoords;

#include "common.glsl"
#include "alpha_test.glsl"

//uniform vec3 diffuseColor;

flat in int fsDrawID;

//uniform float gamma = 2.2;

layout (location = 0) out vec3 color;
layout (location = 1) out vec2 velocity;

// Unjittered
in vec4 fsCurrentClipPos;
in vec4 fsPrevClipPos;

void main() {
    Material material = materials[materialIndices[fsDrawID]];
    vec4 diffuse = FLOAT4_TO_VEC4(material.diffuseColor);
    if (bitwiseAndBool(material.flags, GPU_DIFFUSE_MAPPED)) {
        diffuse = texture(material.diffuseMap, fsTexCoords);
    }

    runAlphaTest(diffuse.a);

    // Apply gamma correction
    //texColor = pow(texColor, vec3(1.0 / gamma));
    color = diffuse.rgb;
    velocity = calculateVelocity(fsCurrentClipPos, fsPrevClipPos);

    // Small offset to help prevent z fighting in certain cases
    if (diffuse.a < 1.0) {
        gl_FragDepth = gl_FragCoord.z - ALPHA_DEPTH_OFFSET;
    }
    else {
        gl_FragDepth = gl_FragCoord.z;
    }
}