#version 410 core

layout (location = 0)  in vec3 position;
layout (location = 1)  in vec2 texCoords;
layout (location = 2)  in vec3 normal;
layout (location = 3)  in vec3 tangent;
layout (location = 4)  in vec3 bitangent;
layout (location = 8)  in vec3 diffuseColor;
layout (location = 9)  in vec3 baseReflectivity;
layout (location = 10) in float metallic;
layout (location = 11) in float roughness;
layout (location = 12) in mat4 model;

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
out float fsRoughness;
out mat4 fsModel;
out mat3 fsModelNoTranslate;
out vec3 fsBaseReflectivity;
out float fsMetallic;
out vec3 fsDiffuseColor;

void main() {
    //mat4 model = modelMats[gl_InstanceID];
    vec4 pos = model * vec4(position, 1.0);
    vec4 viewSpacePos = view * pos;
    fsPosition = pos.xyz;
    fsViewSpacePos = viewSpacePos.xyz;
    fsTexCoords = texCoords;
    fsModelNoTranslate = mat3(model);
    fsNormal = normalize(fsModelNoTranslate * normal);
    // @see https://learnopengl.com/Advanced-Lighting/Normal-Mapping
    // tbn matrix transforms from normal map space to world space
    mat3 normalMatrix = mat3(model);
    vec3 n = normalize(normalMatrix * normal);
    vec3 t = normalize(normalMatrix * tangent);
    // re-orthogonalize T with respect to N - see end of https://learnopengl.com/Advanced-Lighting/Normal-Mapping
    // this is also called Graham-Schmidt
    t = normalize(t - dot(t, n) * n);
    // then retrieve perpendicular vector B with the cross product of T and N
    vec3 b = normalize(cross(n, t));
    fsTbnMatrix = mat3(t, b, n);
    fsRoughness = roughness;
    fsModel = model;
    fsBaseReflectivity = baseReflectivity;
    fsMetallic = metallic;
    fsDiffuseColor = diffuseColor;
    gl_Position = projection * viewSpacePos;
}