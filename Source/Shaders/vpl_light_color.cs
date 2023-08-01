STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

// This defines a 1D local work group of 1 (y and z default to 1)
// See the Compute section of the OpenGL Superbible for more information
//
// Also see https://medium.com/@daniel.coady/compute-shaders-in-opengl-4-3-d1c741998c03
// Also see https://learnopengl.com/Guest-Articles/2022/Compute-Shaders/Introduction
layout (local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

#include "pbr.glsl"
#include "vpl_common.glsl"

uniform vec3 infiniteLightColor;
uniform sampler2DArray infiniteLightDepthMap;
uniform mat4 invCascadeProjViews[4];

// for vec2 with std140 it always begins on a 2*4 = 8 byte boundary
// for vec3, vec4 with std140 it always begins on a 4*4=16 byte boundary

// See https://www.khronos.org/opengl/wiki/Interface_Block_(GLSL)#Memory_layout
// for information on layout std140 and std430. In std140 floats aren't guaranteed
// to be packed and so float arrays in OpenGL are not the same as float arrays in C/C++.
//
// This changes with std430 where it enforces equivalency between OpenGL and C/C++ float arrays
// by tightly packing them.
layout (std430, binding = 0) buffer inoutBlock1 {
    VplData lightData[];
};

layout (std430, binding = 1) readonly buffer inputBlock1 {
    int numVisible[];
};

uniform samplerCubeArray diffuseCubeMaps[MAX_TOTAL_SHADOW_ATLASES];
uniform samplerCubeArray shadowCubeMaps[MAX_TOTAL_SHADOW_ATLASES];

layout (std430, binding = 4) readonly buffer inputBlock3 {
    AtlasEntry diffuseIndices[];
};

vec3 calculateInfiniteWorldSpacePos(vec3 worldPos, vec3 cascadeBlends) {
    vec4 position = vec4(worldPos, 1.0);

    vec3 p1, p2;
    vec3 cascadeCoords[4];
    // cascadeCoords[0] = cascadeCoord0 * 0.5 + 0.5;
    for (int i = 0; i < 4; ++i) {
        // cascadeProjViews[i] * position puts the coordinates into clip space which are on the range of [-1, 1].
        // Since we are looking for texture coordinates on the range [0, 1], we first perform the perspective divide
        // and then perform * 0.5 + vec3(0.5).
        vec4 coords = cascadeProjViews[i] * position;
        cascadeCoords[i] = coords.xyz / coords.w; // Perspective divide
        cascadeCoords[i].xyz = cascadeCoords[i].xyz * 0.5 + vec3(0.5);
    }

    bool beyondCascade2 = cascadeBlends.y >= 0.0;
    bool beyondCascade3 = cascadeBlends.z >= 0.0;
    // p1.z = float(beyondCascade2) * 2.0;
    // p2.z = float(beyondCascade3) * 2.0 + 1.0;

    int index1 = beyondCascade2 ? 2 : 0;
    int index2 = beyondCascade3 ? 3 : 1;
    p1.z = float(index1);
    p2.z = float(index2);

    vec2 depthCoord1 = cascadeCoords[index1].xy;
    vec2 depthCoord2 = cascadeCoords[index2].xy;

    //vec3 blend = saturate(vec3(cascadeBlend[0], cascadeBlend[1], cascadeBlend[2]));
    float weight = beyondCascade2 ? saturate(cascadeBlends.y) - saturate(cascadeBlends.z) : 1.0 - saturate(cascadeBlends.x);

    vec2 wh = computeTexelSize(infiniteLightShadowMap, 0);
                         
    p1.xy = depthCoord1;
    p2.xy = depthCoord2;

    float depth1 = texture(infiniteLightDepthMap, p1).r;
    float depth2 = texture(infiniteLightDepthMap, p2).r;

    vec3 pos1 = worldPositionFromDepth(p1.xy, depth1, invCascadeProjViews[index1]);
    vec3 pos2 = worldPositionFromDepth(p2.xy, depth2, invCascadeProjViews[index2]);

    // blend and return
    //return mix(pos2, pos1, weight);
    return pos1;
}

void main() {
    int stepSize = int(gl_NumWorkGroups.x * gl_WorkGroupSize.x);

    float colorMultiplier = 50000.0;//clamp(float(numVisible) / float(MAX_TOTAL_VPLS_PER_FRAME), 0.1, 1.0) * 500.0;

    int visibleVpls = numVisible[0];

    for (int i = int(gl_GlobalInvocationID.x); i < visibleVpls; i += stepSize) {
        int index = i;
        VplData data = lightData[index];
        AtlasEntry entry = diffuseIndices[index];

        // Calculate new world space pos to be used for indirect specular reflections
        // vec3 cascadeBlends = vec3(
        //     dot(cascadePlanes[0], vec4(data.position.xyz, 1.0)),
        //     dot(cascadePlanes[1], vec4(data.position.xyz, 1.0)),
        //     dot(cascadePlanes[2], vec4(data.position.xyz, 1.0))
        // );

        // vec4 specularPos = vec4(calculateInfiniteWorldSpacePos(data.position.xyz, cascadeBlends), 1.0);

        // First two samples from the exact direction vector for a total of 10 samples after loop
        vec3 color = textureLod(diffuseCubeMaps[entry.index], vec4(-infiniteLightDirection, float(entry.layer)), 0).rgb * infiniteLightColor;
        float magnitude = data.radius * textureLod(shadowCubeMaps[entry.index], vec4(-infiniteLightDirection, float(entry.layer)), 0).r;
        float offset = 0.5;
        float offsets[2] = float[](-offset, offset);
        float totalColors = 1.0;
        // This should result in 2*2*2 = 8 samples, + 2 from above = 10
        // for (int x = 0; x < 2; ++x) {
        //     for (int y = 0; y < 2; ++y) {
        //         for (int z = 0; z < 2; ++z) {
        //             vec3 dirOffset = vec3(offsets[x], offsets[y], offsets[z]);
        //             color += textureLod(diffuseCubeMaps[entry.index], vec4(-infiniteLightDirection + dirOffset, float(entry.layer)), 0).rgb * infiniteLightColor;
        //             totalColors += 1.0;
        //         }
        //     }
        // }

        //vec4 specularPos = data.position;
        vec4 specularPos = vec4(data.position.xyz - magnitude * infiniteLightDirection, 1.0);

        lightData[index].color = vec4(color / totalColors * data.intensity * colorMultiplier, 1.0);
        lightData[index].specularPosition = specularPos;
    }
}