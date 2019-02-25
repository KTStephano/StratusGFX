#pragma once

#include <utility>
#include "glm/glm.hpp"

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
    float uvDet = 1 / (deltaUV1.x * deltaUV2.y - deltaUV1.y * deltaUV2.x);

    // Calculate all values of the inverse of the UV matrix
    float uvA = uvDet * deltaUV2.y;
    float uvB = -uvDet * deltaUV1.y;
    float uvC = -uvDet * deltaUV2.x;
    float uvD = uvDet * deltaUV1.x;

    // Calculate the tangent/bitangent
    glm::vec3 tangent;
    tangent.x = uvA * e1.x + uvB * e2.x;
    tangent.y = uvA * e1.y + uvB * e2.y;
    tangent.z = uvA * e1.z + uvB * e2.z;

    glm::vec3 bitangent;
    bitangent.x = uvC * e1.x + uvD * e2.x;
    bitangent.y = uvC * e1.y + uvD * e2.y;
    bitangent.z = uvC * e1.z + uvD * e2.z;

    return std::make_pair(std::move(tangent), std::move(bitangent));
}