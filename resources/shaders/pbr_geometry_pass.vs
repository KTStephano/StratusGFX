STRATUS_GLSL_VERSION

#include "mesh_data.glsl"

uniform mat4 model;
uniform mat4 projection;
uniform mat4 view;

/**
 * Information about the camera
 */
uniform vec3 viewPosition;

smooth out vec3 fsPosition;
smooth out vec3 fsViewSpacePos;
out vec3 fsNormal;
smooth out vec2 fsTexCoords;

// Made using the tangent, bitangent and normal
out mat3 fsTbnMatrix;
out mat4 fsModel;
out mat3 fsModelNoTranslate;

void main() {
    //mat4 model = modelMats[gl_InstanceID];
    vec4 pos = model * vec4(getPosition(gl_VertexID), 1.0);

    vec4 viewSpacePos = view * pos;
    fsPosition = pos.xyz;
    fsViewSpacePos = viewSpacePos.xyz;
    fsTexCoords = getTexCoord(gl_VertexID);

    fsModelNoTranslate = mat3(model);
    fsNormal = normalize(fsModelNoTranslate * getNormal(gl_VertexID));

    // @see https://learnopengl.com/Advanced-Lighting/Normal-Mapping
    // tbn matrix transforms from normal map space to world space
    mat3 normalMatrix = mat3(model);
    vec3 n = normalize(normalMatrix * getNormal(gl_VertexID));
    vec3 t = normalize(normalMatrix * getTangent(gl_VertexID));
    // re-orthogonalize T with respect to N - see end of https://learnopengl.com/Advanced-Lighting/Normal-Mapping
    // this is also called Graham-Schmidt
    t = normalize(t - dot(t, n) * n);
    // then retrieve perpendicular vector B with the cross product of T and N
    vec3 b = normalize(cross(n, t));
    fsTbnMatrix = mat3(t, b, n);

    fsModel = model;
    
    gl_Position = projection * viewSpacePos;
}