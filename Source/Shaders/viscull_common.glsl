STRATUS_GLSL_VERSION

#include "aabb.glsl"

// See https://stackoverflow.com/questions/5254838/calculating-distance-between-a-point-and-a-rectangular-box-nearest-point
float distanceFromPointToAABB(in vec3 vmin, in vec3 vmax, in vec3 point) {
    float dx = max(vmin.x - point.x, max(0.0, point.x - vmax.x));
    float dy = max(vmin.y - point.y, max(0.0, point.y - vmax.y));
    float dz = max(vmin.z - point.z, max(0.0, point.z - vmax.z));

    return sqrt(dx * dx + dy * dy + dz + dz);
}

float distanceFromPointToAABB(in AABB aabb, in vec3 point) {
    return distanceFromPointToAABB(aabb.vmin.xyz, aabb.vmax.xyz, point);
}