#pragma once

#include <utility>
#include "glm/glm.hpp"

namespace stratus {
// first: tangent, second: bitangent
typedef std::pair<glm::vec3, glm::vec3> tan_bitan_t;

/**
 * Given 3 points and 3 texture coordinates, this calculates
 * the tangent and bitangent. This is especially useful for 
 * performing bump mapping where normal maps need to be transformed
 * into tangent space before calculations are performed.
 *
 * @see https://learnopengl.com/Advanced-Lighting/Normal-Mapping
 */
inline tan_bitan_t calculateTangentAndBitangent(
    glm::vec3 p1, glm::vec3 p2, glm::vec3 p3,
    glm::vec2 uv1, glm::vec2 uv2, glm::vec2 uv3) {

    // Calculate reference lines E1 and E2
    glm::vec3 e1 = p2 - p1;
    glm::vec3 e2 = p3 - p1;

    // Calculate the change in the uv coordinates
    // from one point to another
    glm::vec2 deltaUV1 = uv2 - uv1;
    glm::vec2 deltaUV2 = uv3 - uv1;

    // Compute the determinant
    float uvDet = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV1.y * deltaUV2.x);

    glm::vec3 tangent;
    tangent.x = uvDet * (deltaUV2.y * e1.x - deltaUV1.y * e2.x);
    tangent.y = uvDet * (deltaUV2.y * e1.y - deltaUV1.y * e2.y);
    tangent.z = uvDet * (deltaUV2.y * e1.z - deltaUV1.y * e2.z);
    tangent = glm::normalize(tangent);

    glm::vec3 bitangent;
    bitangent.x = uvDet * (-deltaUV2.x * e1.x + deltaUV1.x * e2.x);
    bitangent.y = uvDet * (-deltaUV2.x * e1.y + deltaUV1.x * e2.y);
    bitangent.z = uvDet * (-deltaUV2.x * e1.z + deltaUV1.x * e2.z);
    bitangent = glm::normalize(bitangent);

    // Calculate all values of the inverse of the UV matrix
    /*
    double uvA = uvDet * deltaUV2.y;
    double uvB = -uvDet * deltaUV1.y;
    double uvC = -uvDet * deltaUV2.x;
    double uvD = uvDet * deltaUV1.x;

    // Calculate the tangent/bitangent
    glm::vec3 tangent;
    tangent.x = float(uvA * e1.x + uvB * e2.x);
    tangent.y = float(uvA * e1.y + uvB * e2.y);
    tangent.z = float(uvA * e1.z + uvB * e2.z);
    tangent = glm::normalize(tangent);

    glm::vec3 bitangent;
    bitangent.x = float(uvC * e1.x + uvD * e2.x);
    bitangent.y = float(uvC * e1.y + uvD * e2.y);
    bitangent.z = float(uvC * e1.z + uvD * e2.z);
    bitangent = glm::normalize(bitangent);
    */

    return std::make_pair(std::move(tangent), std::move(bitangent));
}
}