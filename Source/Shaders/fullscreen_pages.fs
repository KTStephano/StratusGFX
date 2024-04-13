STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require
#extension GL_ARB_sparse_texture2 : require

in vec2 fsTexCoords;

#include "vsm_common.glsl"

uniform sampler2DArray depth;

uniform float znear;
uniform float zfar;

out vec4 color;

// float linearizeDepth(in vec2 uv)
// {
//     uint pageId;
//     uint dirtyBit;
//     uint px;
//     uint py;
//     uint mem;
//     uint unused;

//     int pageFlatIndex = 46 + 7 * int(64);
//     unpackPageMarkerData(
//         currFramePageResidencyTable[pageFlatIndex].info, 
//         unused,
//         px,
//         py,
//         mem,
//         unused,
//         unused
//     );

//     vec2 pixelCoordsCompare = vec2(128 * px, 128 * py);

//     uvec4 value;
//     ivec3 pixelCoords = ivec3(uv * (textureSize(depth, 0).xy - vec2(1.0)), 0.0);
//     // if (pixelCoords.x < pixelCoordsCompare.x || pixelCoords.x > pixelCoordsCompare.x + 128 ||
//     //     pixelCoords.y < pixelCoordsCompare.y || pixelCoords.y > pixelCoordsCompare.y + 128) {
//     //         return 0.0;
//     //     }
//     //ivec3 pixelCoords = ivec3(uv * vec2(127), 0.0) + ivec3(pixelCoordsCompare, 0);
//     int status = sparseTexelFetchARB(depth, pixelCoords, 0, value);
//     float depth = uintBitsToFloat(value.r);
//     if (sparseTexelsResidentARB(status) == false) {
//         return 0.0;
//     }
//     //return depth == 1.0 ? 1.0 : 0.0;
//     // float n = znear;
//     // float f = zfar;
//     // return (2.0 * n) / (f + n - depth * (f - n));
//     return depth;
// }

// See http://glampert.com/2014/01-26/visualizing-the-depth-buffer/
float linearizeDepth(in vec2 uv)
{
    uvec4 value;
    ivec3 pixelCoords = ivec3(uv * (textureSize(depth, 0).xy - vec2(1.0)), 0.0);
    int status = sparseTexelFetchARB(depth, pixelCoords, 0, value);
    float depth = uintBitsToFloat(value.r);
    if (sparseTexelsResidentARB(status) == false) {
        return 0.0;
    }
    //return depth == 1.0 ? 1.0 : 0.0;
    //return (2.0 * znear) / (zfar + znear - depth * (zfar - znear));
    return depth;
}

// void main() {
//     // vec4 value;
//     // int status = sparseTextureARB(depth, vec3(fsTexCoords, 0.0), value);
//     // //color = vec4(vec3(texture(depth, vec3(fsTexCoords, 0.0)).r), 1.0);
//     // vec3 result = vec3(sparseTexelsResidentARB(status) == true ? 1.0 : 0.0, 0.0, 0.0);
//     // color = vec4(result, 1.0);

//     color = vec4(vec3(linearizeDepth(fsTexCoords)), 1.0);
// }

void main() {
    uvec2 pageGroupCoords = uvec2(fsTexCoords * (vec2(64) - vec2(1.0)));
    uint pageGroupIndex = pageGroupCoords.x + pageGroupCoords.y * 64 + 0 * 64 * 64;

    uint pageId;
    uint dirtyBit;
    uint px;
    uint py;
    uint mem;
    uint residencyStatus;
    uint unused;

    //int pageFlatIndex = 46 + 7 * int(64);
    unpackPageMarkerData(
        currFramePageResidencyTable[pageGroupIndex].info, 
        unused,
        px,
        py,
        mem,
        residencyStatus,
        unused
    );

    if (pageGroupIndex != (46 + 7 * int(64))) {
        residencyStatus = 0;
    }

    //color = vec4((value > 0 ? vec3(129.0 / 255.0, 1.0, 104.0 / 255.0) : vec3(1.0, 97.0 / 255.0, 97.0 / 255.0)), 1.0);
    // color = vec4((residencyStatus > 0 ? vec3(129.0 / 255.0, 1.0, 104.0 / 255.0) : vec3(0.0)), 1.0);
    color = vec4((residencyStatus > 0 ? vec3(px / float(64), py / float(64), 1.0) : vec3(0.0)), 1.0);
}