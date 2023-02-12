STRATUS_GLSL_VERSION

#include "mesh_data.glsl"

layout (std430, binding = 13) readonly buffer SSBO3 {
    mat4 modelMatrices[];
};

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

flat out int fsDrawID;

void main() {
    //mat4 model = modelMats[gl_InstanceID];
    vec4 pos = modelMatrices[gl_DrawID] * vec4(getPosition(gl_VertexID), 1.0);

    vec4 viewSpacePos = view * pos;
    fsPosition = pos.xyz;
    fsViewSpacePos = viewSpacePos.xyz;
    fsTexCoords = getTexCoord(gl_VertexID);

    fsModelNoTranslate = mat3(modelMatrices[gl_DrawID]);
    fsNormal = normalize(fsModelNoTranslate * getNormal(gl_VertexID));

    // @see https://learnopengl.com/Advanced-Lighting/Normal-Mapping
    // tbn matrix transforms from normal map space to world space
    mat3 normalMatrix = mat3(modelMatrices[gl_DrawID]);
    vec3 n = normalize(normalMatrix * getNormal(gl_VertexID));
    vec3 t = normalize(normalMatrix * getTangent(gl_VertexID));
    // re-orthogonalize T with respect to N - see end of https://learnopengl.com/Advanced-Lighting/Normal-Mapping
    // this is also called Graham-Schmidt
    t = normalize(t - dot(t, n) * n);
    // then retrieve perpendicular vector B with the cross product of T and N
    vec3 b = normalize(cross(n, t));
    fsTbnMatrix = mat3(t, b, n);

    fsModel = modelMatrices[gl_DrawID];

    fsDrawID = gl_DrawID;
    
    gl_Position = projection * viewSpacePos;
}