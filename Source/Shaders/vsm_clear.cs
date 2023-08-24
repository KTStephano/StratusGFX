STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require
#extension GL_ARB_sparse_texture2 : require

layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

precision highp float;
precision highp int;
precision highp uimage2D;

#include "common.glsl"
#include "vsm_common.glsl"

layout (r32ui) coherent uniform uimage2DArray vsm;

layout (std430, binding = 4) buffer block1 {
    PageResidencyEntry currFramePageResidencyTable[];
};

uniform float clearValue = 1.0;

uniform mat4 invCascadeProjectionView;
uniform mat4 vsmProjectionView;

uniform ivec2 startXY;
uniform ivec2 endXY;

uniform ivec2 numPagesXY;

shared uint clearValueBits;
shared ivec2 vsmSize;
shared ivec2 vsmMaxIndex;

void main() {
    ivec2 virtualPixelCoords = ivec2(gl_GlobalInvocationID.xy + startXY);

    if (gl_LocalInvocationID == 0) {
        clearValueBits = floatBitsToUint(clearValue);
        vsmSize = imageSize(vsm).xy;
        vsmMaxIndex = vsmSize - ivec2(1.0);
    }

    barrier();

    if (virtualPixelCoords.x < endXY.x && virtualPixelCoords.y < endXY.y) {
        vec2 physicalPixelCoords = convertVirtualCoordsToPhysicalCoords(virtualPixelCoords, vsmMaxIndex, invCascadeProjectionView, vsmProjectionView);

        imageStore(vsm, ivec3(round(physicalPixelCoords), 0), uvec4(clearValueBits));
        //imageStore(vsm, ivec3(floor(physicalPixelCoords), 0), uvec4(clearValueBits));
        // imageStore(vsm, ivec3(floor(physicalPixelCoords), 0), uvec4(clearValueBits));
        // imageStore(vsm, ivec3(ceil(physicalPixelCoords), 0), uvec4(clearValueBits));

	    // imageStore(vsm, ivec3(round(physicalPixelCoords), 0) + ivec3(0, 1, 0), uvec4(clearValueBits));
	    // imageStore(vsm, ivec3(round(physicalPixelCoords), 0) + ivec3(0, -1, 0), uvec4(clearValueBits));
	    // imageStore(vsm, ivec3(round(physicalPixelCoords), 0) + ivec3(1, 0, 0), uvec4(clearValueBits));
	    // imageStore(vsm, ivec3(round(physicalPixelCoords), 0) + ivec3(-1, 0, 0), uvec4(clearValueBits));

        // TODO: Figure out why dividing physicalPixelCoords by numPagesXY gave the wrong answer
        vec2 physicalPageTexCoords = vec2(physicalPixelCoords) / vec2(vsmMaxIndex);
        ivec2 physicalPageCoords = ivec2(round(physicalPageTexCoords * (vec2(numPagesXY) - vec2(1.0))));
        atomicAnd(currFramePageResidencyTable[physicalPageCoords.x + physicalPageCoords.y * numPagesXY.x].info, VSM_PAGE_ID_MASK);
    }
}