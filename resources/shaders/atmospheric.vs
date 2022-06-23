#version 410 core

uniform vec3 frustumParams; // aspect ratio / projection dist, 1.0 / projection dist, dmin (znear)
uniform mat4 shadowMatrix;  // M_shadow(0) * M_camera, aka transform from camera space -> shadow space for cascade 0

layout (location = 0)  in vec3 position;
layout (location = 1)  in vec2 texCoords;
layout (location = 2)  in vec3 normal;

smooth out vec2 fsTexCoords;
smooth out vec2 fsCamSpaceRay;
smooth out vec3 fsShadowSpaceRay;

// Calculates q from page 342, eq. 10.61
vec3 calculateCameraSpaceRayDirection(vec3 ndcVertex) {
    return frustumParams.z * vec3(frustumParams.x * ndcVertex.x, frustumParams.y * ndcVertex.y, 1.0);
}

// Calculates r from page 342, eq 10.62
vec3 calculateShadowSpaceRayDirection(vec3 cameraSpaceRayDirection) {
    mat3 shadowMatNoTransform = mat3(shadowMatrix);
    return shadowMatNoTransform * cameraSpaceRayDirection;
}

void main() {
    fsTexCoords = texCoords;

    vec3 q = calculateCameraSpaceRayDirection(position);
    vec3 r = calculateShadowSpaceRayDirection(q);
    fsCamSpaceRay = q.xy;
    fsShadowSpaceRay = r;

    gl_Position = vec4(position, 1.0);
}