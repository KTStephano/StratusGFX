STRATUS_GLSL_VERSION

// Axis-Aligned Bounding Box
// See the section on visibility and culling in "Foundations of Game Engine Development, Volume 2: Rendering"
// Also see the section on frustum culling in "3D Graphics Rendering Cookbook"
struct AABB {
    vec4 vmin;
    vec4 vmax;
    //vec4 center;
    //vec4 size;
};

uniform vec4 frustumPlanes[6];
uniform vec4 frustumCorners[8];

vec4[24] convertCornersToLineVertices(in vec4 corners[8]) {
    vec4 vertices[24] = vec4[](
        corners[0], corners[1],
        corners[2], corners[3],
        corners[4], corners[5],
        corners[6], corners[7],

        corners[0], corners[2],
        corners[1], corners[3],
        corners[4], corners[6],
        corners[5], corners[7],

        corners[0], corners[4],
        corners[1], corners[5],
        corners[2], corners[6],
        corners[3], corners[7]
    );

    return vertices;
}

vec4[8] computeCornersWithTransform(in AABB aabb, in mat4 transform) {
    vec4 vmin = aabb.vmin;
    vec4 vmax = aabb.vmax;

    vec4 corners[8] = vec4[](
        transform * vec4(vmin.x, vmin.y, vmin.z, 1.0),
        transform * vec4(vmin.x, vmax.y, vmin.z, 1.0),
        transform * vec4(vmin.x, vmin.y, vmax.z, 1.0),
        transform * vec4(vmin.x, vmax.y, vmax.z, 1.0),
        transform * vec4(vmax.x, vmin.y, vmin.z, 1.0),
        transform * vec4(vmax.x, vmax.y, vmin.z, 1.0),
        transform * vec4(vmax.x, vmin.y, vmax.z, 1.0),
        transform * vec4(vmax.x, vmax.y, vmax.z, 1.0)
        /*
        transform * vec4(vmin.x, vmin.y, vmin.z, 1.0),
        transform * vec4(vmin.x, vmin.y, vmax.z, 1.0),
        transform * vec4(vmin.x, vmax.y, vmin.z, 1.0),
        transform * vec4(vmax.x, vmin.y, vmin.z, 1.0),
        transform * vec4(vmin.x, vmax.y, vmax.z, 1.0),
        transform * vec4(vmax.x, vmin.y, vmax.z, 1.0),
        transform * vec4(vmax.x, vmax.y, vmin.z, 1.0),
        transform * vec4(vmax.x, vmax.y, vmax.z, 1.0)
        */
    );

    return corners;
}

// This code was taken from "3D Graphics Rendering Cookbook" source code, shared/UtilsMath.h
AABB transformAabb(in AABB aabb, in mat4 transform) {
    vec4 corners[8] = computeCornersWithTransform(aabb, transform);

    vec3 vmin3 = corners[0].xyz;
    vec3 vmax3 = corners[0].xyz;

    for (int i = 1; i < 8; ++i) {
        vmin3 = min(vmin3, corners[i].xyz);
        vmax3 = max(vmax3, corners[i].xyz);
    }

    AABB result;
    result.vmin = vec4(vmin3, 1.0);
    result.vmax = vec4(vmax3, 1.0); 

    return result;
}

bool isAabbInFrustum(in vec4 vmin, in vec4 vmax) {    
    for (int i = 0; i < 6; ++i) {
        int r = 0;
        r += (dot(frustumPlanes[i], vec4(vmin.x, vmin.y, vmin.z, 1.0f)) < 0) ? 1 : 0;
        r += (dot(frustumPlanes[i], vec4(vmax.x, vmin.y, vmin.z, 1.0f)) < 0) ? 1 : 0; 
        r += (dot(frustumPlanes[i], vec4(vmin.x, vmax.y, vmin.z, 1.0f)) < 0) ? 1 : 0; 
        r += (dot(frustumPlanes[i], vec4(vmax.x, vmax.y, vmin.z, 1.0f)) < 0) ? 1 : 0; 
        r += (dot(frustumPlanes[i], vec4(vmin.x, vmin.y, vmax.z, 1.0f)) < 0) ? 1 : 0; 
        r += (dot(frustumPlanes[i], vec4(vmax.x, vmin.y, vmax.z, 1.0f)) < 0) ? 1 : 0; 
        r += (dot(frustumPlanes[i], vec4(vmin.x, vmax.y, vmax.z, 1.0f)) < 0) ? 1 : 0; 
        r += (dot(frustumPlanes[i], vec4(vmax.x, vmax.y, vmax.z, 1.0f)) < 0) ? 1 : 0; 

        if (r == 8) return false;
    }

    return true;
}

bool isFrustumInAabb(in vec4 vmin, in vec4 vmax) {
    int r;

    r = 0;
    for (int i = 0; i < 8; ++i) {
        r += ((frustumCorners[i].x > vmax.x) ? 1 : 0);
    }
    if (r == 8) return false;

    r = 0;
    for (int i = 0; i < 8; ++i) {
        r += ((frustumCorners[i].x < vmin.x) ? 1 : 0);
    }
    if (r == 8) return false;

    r = 0;
    for (int i = 0; i < 8; ++i) {
        r += ((frustumCorners[i].y > vmax.y) ? 1 : 0);
    }
    if (r == 8) return false;

    r = 0;
    for (int i = 0; i < 8; ++i) {
        r += ((frustumCorners[i].y < vmin.y) ? 1 : 0);
    }
    if (r == 8) return false;

    r = 0;
    for (int i = 0; i < 8; ++i) {
        r += ((frustumCorners[i].z > vmax.z) ? 1 : 0);
    }
    if (r == 8) return false;

    r = 0;
    for (int i = 0; i < 8; ++i) {
        r += ((frustumCorners[i].z < vmin.z) ? 1 : 0);
    }
    if (r == 8) return false;

    return true;
}