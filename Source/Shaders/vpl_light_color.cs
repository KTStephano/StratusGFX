STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require
#extension GL_ARB_gpu_shader_int64 : enable

#include "bindings.glsl"

// This defines a 1D local work group of 1 (y and z default to 1)
// See the Compute section of the OpenGL Superbible for more information
//
// Also see https://medium.com/@daniel.coady/compute-shaders-in-opengl-4-3-d1c741998c03
// Also see https://learnopengl.com/Guest-Articles/2022/Compute-Shaders/Introduction
//
// 8, 8, 6 corresponds to: 7*6*6, the size of a single light probe
// These need to match up with the dimensions of a light probe defined in the C++ code
layout (local_size_x = 8, local_size_y = 8, local_size_z = 6) in;

#include "pbr.glsl"
#include "vpl_common.glsl"

uniform vec3 infiniteLightColor;
uniform float infiniteLightIntensity;
uniform int visibleVpls;

// for vec2 with std140 it always begins on a 2*4 = 8 byte boundary
// for vec3, vec4 with std140 it always begins on a 4*4=16 byte boundary

// See https://www.khronos.org/opengl/wiki/Interface_Block_(GLSL)#Memory_layout
// for information on layout std140 and std430. In std140 floats aren't guaranteed
// to be packed and so float arrays in OpenGL are not the same as float arrays in C/C++.
//
// This changes with std430 where it enforces equivalency between OpenGL and C/C++ float arrays
// by tightly packing them.
// layout (std430, binding = 1) readonly buffer inputBlock1 {
//     int numVisible[];
// };

#define MAX_IMAGES_PER_BATCH (2)

// Only 8 images bound total are allowed by OpenGL
//layout (rgba8) readonly uniform imageCubeArray diffuseCubeMaps[MAX_IMAGES_PER_BATCH];
//layout (rgba32f) readonly uniform imageCubeArray positionCubeMaps[MAX_IMAGES_PER_BATCH];
//layout (rgba16f) coherent uniform imageCubeArray lightingCubeMaps[MAX_IMAGES_PER_BATCH];

layout (std430, binding = VPL_DIFFUSE_CUBE_IMAGES) readonly buffer imageBlock1 {
    uint64_t diffuseCubeMaps[];
};

layout (std430, binding = VPL_POSITION_CUBE_IMAGES) readonly buffer imageBlock2 {
    uint64_t positionCubeMaps[];
};

layout (std430, binding = VPL_LIGHTING_CUBE_IMAGES) readonly buffer imageBlock3 {
    uint64_t lightingCubeMaps[];
};

layout (std430, binding = VPL_PROBE_DATA_BINDING) buffer inputBlock1 {
    VplData probes[];
};

layout (std430, binding = 4) readonly buffer inputBlock3 {
    AtlasEntry diffuseIndices[];
};

//layout (std430, binding = VPL_PROBE_CONTRIB_BINDING) writeonly buffer inputBlock4 {
//    int probeFlags[];
//};

// We pass them in as uint64_t and then cast them to the appropriate image type to get around
// the limitation of only allowing 8 bound images
// layout (std430, binding = 5) readonly buffer inputBlock4 {
//     uint64_t lightingCubeHandles[];
// };

// See https://stackoverflow.com/questions/13892732/texelfetch-from-cubemap
// See https://stackoverflow.com/questions/6980530/selecting-the-face-of-a-cubemap-in-glsl
vec3 generateCubemapCoords(in vec2 txc, in int face) {
  vec3 v;
  switch(face) {
    case 0: v = vec3( 1.0, -txc.x, txc.y); break; // +X
    case 1: v = vec3(-1.0,  txc.x, txc.y); break; // -X
    case 2: v = vec3( txc.x,  1.0, txc.y); break; // +Y
    case 3: v = vec3(-txc.x, -1.0, txc.y); break; // -Y
    case 4: v = vec3(txc.x, -txc.y,  1.0); break; // +Z
    case 5: v = vec3(txc.x,  txc.y, -1.0); break; // -Z
  }
//   switch(face) {
//     case 0: v = vec3( 1.0,  txc.x, txc.y); break; // +X
//     case 1: v = vec3(-1.0,  txc.x, txc.y); break; // -X
//     case 2: v = vec3(txc.x,  1.0,  txc.y); break; // +Y
//     case 3: v = vec3(txc.x, -1.0,  txc.y); break; // -Y
//     case 4: v = vec3(txc.x, txc.y,  1.0); break;  // +Z
//     case 5: v = vec3(txc.x, txc.y, -1.0); break;  // -Z
//   }
  return normalize(v);
}

shared int currentProbeIsVisible;
shared int numNonDirectLightSamples;
shared int minDistance;
shared uint diffuseXDirect, diffuseYDirect, diffuseZDirect;
shared uint diffuseXSky, diffuseYSky, diffuseZSky;

void main() {
    // Smaller than normal since we want to reduce simulated 2nd bounce strength
    const int probeRadius = 250;    

    // From all faces
    const int totalTexelSamples = 8*8*6;

    // Pulls from the value set by glDispatchCompute
    int stepSize = int(gl_NumWorkGroups.x);
    //int visibleVpls = numVisible[0];

    if (gl_LocalInvocationIndex == 0) {
        // Set state
        currentProbeIsVisible = 0;
        numNonDirectLightSamples = 0;
        minDistance = probeRadius;
        diffuseXDirect = 0, diffuseYDirect = 0, diffuseZDirect = 0;
        diffuseXSky = 0, diffuseYSky = 0, diffuseZSky = 0;
    }

    barrier();

    // Each work group processes all 6 faces of the light probe
    for (int index = int(gl_WorkGroupID.x); index < visibleVpls; index += stepSize) {
        VplData probe = probes[index];
        if (probe.pendingRelight < 1.0) {
            // All worker threads should see this as well
            continue;
        }

        AtlasEntry entry = diffuseIndices[index];
        // Lighting layer is used specifically to index into lightingCubeMaps
        int lightingIndex = int(entry.index);

        layout (rgba8) imageCubeArray diffuse = layout(rgba8) imageCubeArray(diffuseCubeMaps[lightingIndex]);
        layout (rgba16f) imageCubeArray position = layout (rgba16f) imageCubeArray(positionCubeMaps[lightingIndex]);
        // See https://stackoverflow.com/questions/32349423/create-an-image2d-from-a-uint64-t-image-handle
        //layout (rgba8) imageCubeArray lighting = layout(rgba8) imageCubeArray(lightingCubeHandles[entry.index]);
        layout (rgba16f) imageCubeArray lighting = layout (rgba16f) imageCubeArray(lightingCubeMaps[lightingIndex]);

        // See https://github.com/KhronosGroup/SPIRV-Cross/issues/578
        // cubeArrays need to be thought of as a 2D array, so they only take a 3d index instead of 4d
        ivec3 texelIndex = ivec3(ivec2(gl_LocalInvocationID.xy), int(gl_LocalInvocationID.z)+6*int(entry.layer));

        vec4 diffuseValBase = imageLoad(diffuse, texelIndex).rgba;
        //vec3 diffuseVal = diffuseValBase * infiniteLightColor.rgb;
        vec3 positionVal = imageLoad(position, texelIndex).xyz;

        vec3 lightColor = vec3(0.0);
        float shadowFactor = 1.0;
        if (diffuseValBase.a > 0.0) {
            shadowFactor = 0.0;
            // Signals that we are a sample that doesn't necessarily have to depend on directional light hitting a surface
            atomicAdd(numNonDirectLightSamples, 1);
        } else {
            vec3 cascadeBlends = vec3(dot(cascadePlanes[0], vec4(positionVal, 1.0)),
                            dot(cascadePlanes[1], vec4(positionVal, 1.0)),
                            dot(cascadePlanes[2], vec4(positionVal, 1.0)));
            shadowFactor = 1.0 - calculateInfiniteShadowValue1Sample(vec4(positionVal, 1.0), cascadeBlends, infiniteLightDirection, false);
            //lightColorModifier = infiniteLightIntensity * infiniteLightColor.rgb;
        }

        int distance = max(int(ceil(length(FLOAT3_TO_VEC3(probe.position) - positionVal))), 1);
        if (shadowFactor < 1.0) {
            // Signals to other workers in this group that the light was visible in some way
            atomicAdd(currentProbeIsVisible, 1);
            atomicMin(minDistance, distance);
        }

        barrier();

        // Perform atomic sum of individual direct or sky contributions so we can spread indirect light
        // to neighboring pixels
        //float sampleRatioDirect = 1.0 / (float(currentProbeIsVisible) - float(numNonDirectLightSamples));
        //float sampleRatioSky = 1.0 - sampleRatioDirect;
        float directLightSamples = float(currentProbeIsVisible) - float(numNonDirectLightSamples);
        float sampleRatioDirect = directLightSamples == 0.0 ? 0.0 : 1.0 / float(directLightSamples);
        float sampleRatioSky = numNonDirectLightSamples == 0 ? 0.0 : 1.0 / float(numNonDirectLightSamples);
        if (shadowFactor < 1.0) {
            vec3 lightColorModifier = vec3(1.0);
            vec3 sampleModifier = vec3(1.0);
            uint unused;
            if (diffuseValBase.a > 0.0) {
                sampleModifier = vec3(0.125 * (infiniteLightIntensity) * sampleRatioSky);
                lightColorModifier = vec3(5 + infiniteLightIntensity);

                ATOMIC_ADD_FLOAT(diffuseXSky, diffuseValBase.x * sampleModifier.x, unused)
                ATOMIC_ADD_FLOAT(diffuseYSky, diffuseValBase.y * sampleModifier.y, unused)
                ATOMIC_ADD_FLOAT(diffuseZSky, diffuseValBase.z * sampleModifier.z, unused)
            } else {
                lightColorModifier = infiniteLightIntensity * infiniteLightColor.rgb;
                sampleModifier = lightColorModifier * sampleRatioDirect;

                ATOMIC_ADD_FLOAT(diffuseXDirect, diffuseValBase.x * sampleModifier.x, unused)
                ATOMIC_ADD_FLOAT(diffuseYDirect, diffuseValBase.y * sampleModifier.y, unused)
                ATOMIC_ADD_FLOAT(diffuseZDirect, diffuseValBase.z * sampleModifier.z, unused)
            }

            lightColor = diffuseValBase.rgb * lightColorModifier;
        }

        barrier();

        if (currentProbeIsVisible > 0 && shadowFactor >= 1.0) {
            float weight = 1.0 - (float(distance + minDistance) / float(probeRadius));
            vec3 direct = vec3(uintBitsToFloat(diffuseXDirect), uintBitsToFloat(diffuseYDirect), uintBitsToFloat(diffuseZDirect));// / float(currentProbeIsVisible);
            vec3 sky = vec3(uintBitsToFloat(diffuseXSky), uintBitsToFloat(diffuseYSky), uintBitsToFloat(diffuseZSky));
            vec3 newDiffuseVal = ((diffuseValBase.rgb * direct)) + sky;// * infiniteLightIntensity * infiniteLightColor.rgb;
            lightColor = newDiffuseVal * weight;
        }

        imageStore(lighting, texelIndex, vec4(lightColor, 0.0));

        barrier();

        if (gl_LocalInvocationIndex == 0) {
            // Reset state
            currentProbeIsVisible = 0;
            numNonDirectLightSamples = 0;
            minDistance = probeRadius;
            diffuseXDirect = 0, diffuseYDirect = 0, diffuseZDirect = 0;
            diffuseXSky = 0, diffuseYSky = 0, diffuseZSky = 0;
        }

        barrier();
    }
}