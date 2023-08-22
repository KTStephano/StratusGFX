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

uniform float clearValue = 1.0;

uniform mat4 invCascadeProjectionView;
uniform mat4 vsmProjectionView;

uniform ivec2 startXY;
uniform ivec2 endXY;

shared uint clearValueBits;
shared ivec2 vsmSize;
shared ivec2 vsmMaxIndex;

void main() {
    if (gl_LocalInvocationID == 0) {
        clearValueBits = floatBitsToUint(clearValue);
        vsmSize = imageSize(vsm).xy;
        vsmMaxIndex = vsmSize - ivec2(1.0);
    }

    barrier();

    ivec2 pixelCoords = ivec2(gl_GlobalInvocationID.xy + startXY);

    if (pixelCoords.x < endXY.x && pixelCoords.y < endXY.y) {
        ivec2 physicalPageCoords = ivec2(
            convertVirtualCoordsToPhysicalCoords(pixelCoords, vsmMaxIndex, invCascadeProjectionView, vsmProjectionView)
        );

        imageStore(vsm, ivec3(physicalPageCoords, 0), uvec4(clearValueBits));
    }
}