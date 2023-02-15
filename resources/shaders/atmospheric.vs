STRATUS_GLSL_VERSION

#include "mesh_data.glsl"

uniform vec3 frustumParams; // aspect ratio / projection dist, 1.0 / projection dist, dmin
uniform mat4 shadowMatrix;  // M_shadow(0) * M_camera, aka transform from camera space -> shadow space for cascade 0

smooth out vec2 fsTexCoords;
smooth out vec2 fsCamSpaceRay;
smooth out vec3 fsShadowSpaceRay;

// Calculates q from page 342, eq. 10.61
vec3 calculateCameraSpaceRayDirection(vec3 ndcVertex) {
    return frustumParams.z * vec3(frustumParams.x * ndcVertex.x, frustumParams.y * ndcVertex.y, -1.0);
}

// Calculates r from page 342, eq 10.62
vec3 calculateShadowSpaceRayDirection(vec3 cameraSpaceRayDirection) {
    // We want r to be a direction vector in shadow space
    mat3 shadowMatNoTransform = mat3(shadowMatrix);
    return shadowMatNoTransform * cameraSpaceRayDirection;
}

void main() {
    fsTexCoords = getTexCoord(gl_VertexID);

    vec3 q = calculateCameraSpaceRayDirection(getPosition(gl_VertexID));
    vec3 r = calculateShadowSpaceRayDirection(q);
    fsCamSpaceRay = q.xy;
    fsShadowSpaceRay = r;

    gl_Position = vec4(getPosition(gl_VertexID), 1.0);
}