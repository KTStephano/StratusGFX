STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require
#extension GL_ARB_sparse_texture2 : require

#include "vsm_common.glsl"
#include "bindings.glsl"

in vec2 fsTexCoords;

uniform uint numPageGroupsX;
uniform uint numPageGroupsY;
uniform uint numPagesXY;

out vec4 color;

void main() {
    uvec2 pageGroupCoords = uvec2(fsTexCoords * (vec2(numPagesXY) - vec2(1.0)));
    uint pageGroupIndex = pageGroupCoords.x + pageGroupCoords.y * numPagesXY + 0 * numPagesXY * numPagesXY;

    uint value = pageGroupsToRender[pageGroupIndex];

    //color = vec4((value > 0 ? vec3(129.0 / 255.0, 1.0, 104.0 / 255.0) : vec3(1.0, 97.0 / 255.0, 97.0 / 255.0)), 1.0);
    color = vec4((value > 0 ? vec3(129.0 / 255.0, 1.0, 104.0 / 255.0) : vec3(0.0)), 1.0);
}

// void main() {
//     uvec2 pageCoords = uvec2(fsTexCoords * (vec2(numPagesXY) - vec2(1.0)));
//     uint pageIndex = pageCoords.x + pageCoords.y * numPagesXY + 0 * numPagesXY * numPagesXY;

//     PageResidencyEntry entry = currFramePageResidencyTable[pageIndex];

//     // uint unused1;
//     // uint unused2;
//     // uint unused3;
//     // uint memPool;
//     // uint unused4;
//     // unpackPageMarkerData(entry.frameMarker, unused1, unused2, unused3, memPool, unused4);

//     bool pageValid = unpackFrameMarker(entry.info) > 0;
//     bool pageDirty = unpackDirtyBit(entry.info) > 0;

//     // bool pageValid = pageGroupsToRender[pageIndex] > 0;
//     // bool pageDirty = pageValid;

//     vec3 pageColor = pageValid ? vec3(129.0 / 255.0, 1.0, 104.0 / 255.0) : vec3(0.0);
//     if (pageDirty) {
//         pageColor = vec3(1.0, 97.0 / 255.0, 97.0 / 255.0);
//     }

//     // bool pageValid = memPool == 0;
//     // vec3 pageColor = pageValid ? vec3(129.0 / 255.0, 1.0, 104.0 / 255.0) : vec3(1.0, 0.0, 0.0);

//     color = vec4(pageColor, 1.0);
// }