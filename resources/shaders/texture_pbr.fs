#version 410 core

uniform sampler2D diffuseTexture;

//uniform float fsShininessVals[MAX_INSTANCES];
//uniform float fsShininess = 0.0;
in float fsRoughness;
uniform float heightScale = 0.1;

/**
 * Information about the camera
 */
uniform vec3 viewPosition;

/**
 * Fragment information. All values should be
 * in world space.
 */
in vec3 fsPosition;
in vec3 fsNormal;
in vec2 fsTexCoords;
in mat4 fsModel;
in vec3 fsDiffuseColor;
in vec3 fsBaseReflectivity; // Ex: vec3(0.03-0.04) for plastics
in float fsMetallic; // Between 0 and 1 where 0 is not metallic at all and 1 is purely metallic
//in float fsfsShininess;

/**
 * Tangent space -> world space
 */
in mat3 fsTbnMatrix;

// GBuffer outputs
layout (location = 0) out vec3 gPosition;
layout (location = 1) out vec3 gNormal;
layout (location = 2) out vec3 gAlbedo;
layout (location = 3) out vec3 gBaseReflectivity;
layout (location = 4) out vec3 gRoughnessMetallicAmbient;

void main() {
    vec3 viewDir = normalize(viewPosition - fsPosition);
    vec2 texCoords = fsTexCoords;

    vec3 baseColor = texture(diffuseTexture, texCoords).rgb;
    vec3 normal = fsNormal;
    // Normals generally have values from [-1, 1], but inside
    // an OpenGL texture they are transformed to [0, 1]. To convert
    // them back, we multiply by 2 and subtract 1.
    //normal = normal * 2.0 - 1.0;
    //normal = normalize(fsTbnMatrix * normal);

    // Coordinate space is set to world
    gPosition = fsPosition;
    gNormal = normal;
    gAlbedo = baseColor;
    gBaseReflectivity = fsBaseReflectivity;
    gRoughnessMetallicAmbient = vec3(fsRoughness, fsMetallic, 1.0);
}