#pragma once

#include <utility>
#include "glm/glm.hpp"

namespace stratus {
// first: tangent, second: bitangent
struct TangentBitangent {
    glm::vec3 tangent;
    glm::vec3 bitangent;
};

/**
 * Given 3 points and 3 texture coordinates, this calculates
 * the tangent and bitangent. This is especially useful for 
 * performing bump mapping where normal maps need to be transformed
 * into tangent space before calculations are performed.
 *
 * @see https://learnopengl.com/Advanced-Lighting/Normal-Mapping
 * @see https://marti.works/posts/post-calculating-tangents-for-your-mesh/post/
 */
inline TangentBitangent calculateTangentAndBitangent(
    const glm::vec3 & p1, const glm::vec3 & p2, const glm::vec3 & p3,
    const glm::vec2 & uv1, const glm::vec2 & uv2, const glm::vec2 & uv3) {

    // Calculate reference lines E1 and E2
    glm::vec3 edge1 = p2 - p1;
    glm::vec3 edge2 = p3 - p1;

    // Calculate the change in the uv coordinates
    // from one point to another
    glm::vec2 deltaUV1 = uv2 - uv1;
    glm::vec2 deltaUV2 = uv3 - uv1;

    // Compute the determinant
    float uvDet = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV1.y * deltaUV2.x);

    glm::vec3 tangent(
        ((edge1.x * deltaUV2.y) - (edge2.x * deltaUV1.y)) * uvDet,
        ((edge1.y * deltaUV2.y) - (edge2.y * deltaUV1.y)) * uvDet,
        ((edge1.z * deltaUV2.y) - (edge2.z * deltaUV1.y)) * uvDet
    );

    glm::vec3 bitangent(
        ((edge1.x * deltaUV2.x) - (edge2.x * deltaUV1.x)) * uvDet,
        ((edge1.y * deltaUV2.x) - (edge2.y * deltaUV1.x)) * uvDet,
        ((edge1.z * deltaUV2.x) - (edge2.z * deltaUV1.x)) * uvDet
    );

    return TangentBitangent{std::move(tangent), std::move(bitangent)};
}

static void matRotate(glm::mat4 & out, const glm::vec3 & angles) {
    float angleX = glm::radians(angles.x);
    float angleY = glm::radians(angles.y);
    float angleZ = glm::radians(angles.z);

    float cx = std::cos(angleX);
    float cy = std::cos(angleY);
    float cz = std::cos(angleZ);

    float sx = std::sin(angleX);
    float sy = std::sin(angleY);
    float sz = std::sin(angleZ);

    out[0] = glm::vec4(cy * cz,
                       sx * sy * cz + cx * sz,
                       -cx * sy * cz + sx * sz,
                       out[0].w);

    out[1] = glm::vec4(-cy * sz,
                       -sx * sy * sz + cx * cz,
                       cx * sy * sz + sx * cz,
                       out[1].w);

    out[2] = glm::vec4(sy,
                       -sx * cy,
                       cx * cy, out[2].w);
}

// Inserts a 3x3 matrix into the upper section of a 4x4 matrix
static void matInset(glm::mat4 & out, const glm::mat3 & in) {
    out[0].x = in[0].x;
    out[0].y = in[0].y;
    out[0].z = in[0].z;

    out[1].x = in[1].x;
    out[1].y = in[1].y;
    out[1].z = in[1].z;

    out[2].x = in[2].x;
    out[2].y = in[2].y;
    out[2].z = in[2].z;
}

static void matScale(glm::mat4 & out, const glm::vec3 & scale) {
    out[0].x = out[0].x * scale.x;
    out[0].y = out[0].y * scale.y;
    out[0].z = out[0].z * scale.z;

    out[1].x = out[1].x * scale.x;
    out[1].y = out[1].y * scale.y;
    out[1].z = out[1].z * scale.z;

    out[2].x = out[2].x * scale.x;
    out[2].y = out[2].y * scale.y;
    out[2].z = out[2].z * scale.z;
}

static void matTranslate(glm::mat4 & out, const glm::vec3 & translate) {
    out[3].x = translate.x;
    out[3].y = translate.y;
    out[3].z = translate.z;
}
}