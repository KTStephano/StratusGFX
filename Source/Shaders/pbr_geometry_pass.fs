STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

#include "common.glsl"
#include "alpha_test.glsl"

//uniform float fsShininessVals[MAX_INSTANCES];
//uniform float fsShininess = 0.0;
uniform float heightScale = 0.1;

/**
 * Information about the camera
 */
uniform vec3 viewPosition;

/**
 * Fragment information. All values should be
 * in world space.
 */
smooth in vec3 fsPosition;
//in vec3 fsViewSpacePos;
in vec3 fsNormal;
smooth in vec2 fsTexCoords;
in mat4 fsModel;
in mat3 fsModelNoTranslate;
flat in int fsDrawID;

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
// The structure buffer contains information related to depth in camera space. Useful for things such as ambient occlusion
// and atmospheric shadowing.
layout (location = 5) out vec4 gStructureBuffer;

// See Foundations of Game Engine Development: Volume 2 (The Structure Buffer)
vec4 calculateStructureOutput(float z) {
    // 0xFFFFE000 allows us to extract the upper 10 bits of precision. z - h then
    // Removes the upper 10 bits of precision and leaves us with at least 11 bits of precision.
    //
    // When h and z - h are later recombined, the result will have at least 21 of the original
    // 23 floating point mantissa.
    float h = uintBitsToFloat(floatBitsToUint(z) & 0xFFFFE000U);
    // See https://stackoverflow.com/questions/16365385/explanation-of-dfdx for an explanation of dFd(x|y)
    // They are effectively calculating change in depth between nearest neighbors
    return vec4(dFdx(z), dFdy(z), h, z - h);
}

// See https://learnopengl.com/Advanced-Lighting/Parallax-Mapping
vec2 calculateDepthCoords(in Material material, vec2 texCoords, vec3 viewDir) {
    float height = texture(material.depthMap, texCoords).r;
    vec2 p = viewDir.xy * (height * 0.005);
    return texCoords - p;
}

void main() {
    vec3 viewDir = normalize(viewPosition - fsPosition);
    vec2 texCoords = fsTexCoords;
    Material material = materials[materialIndices[fsDrawID]];

    if (bitwiseAndBool(material.flags, GPU_DEPTH_MAPPED)) {
        texCoords = calculateDepthCoords(material, texCoords, viewDir);
        // if(texCoords.x > 1.0 || texCoords.y > 1.0 || texCoords.x < 0.0 || texCoords.y < 0.0) {
        //     discard;
        // }
    }

    vec4 baseColor = material.diffuseColor;
    vec3 normal = (fsNormal + 1.0) * 0.5; // [-1, 1] -> [0, 1]
    float roughness = material.metallicRoughness.y;
    float ao = 1.0;
    float metallic = material.metallicRoughness.x;

    if (bitwiseAndBool(material.flags, GPU_DIFFUSE_MAPPED)) {
        baseColor = texture(material.diffuseMap, texCoords);
    }

    runAlphaTest(baseColor.a, 0.25);

    if (bitwiseAndBool(material.flags, GPU_NORMAL_MAPPED)) {
        normal = texture(material.normalMap, texCoords).rgb;
        // Normals generally have values from [-1, 1], but inside
        // an OpenGL texture they are transformed to [0, 1]. To convert
        // them back, we multiply by 2 and subtract 1.
        normal = normalize(normal * 2.0 - vec3(1.0)); // [0, 1] -> [-1, 1]
        // fsTbnMatrix goes from tangent space (defined by coordinate system of normal map)
        // to object space, and then model no translate moves to world space without translating
        normal = normalize(fsModelNoTranslate * fsTbnMatrix * normal);
        normal = (normal + vec3(1.0)) * 0.5; // [-1, 1] -> [0, 1]
    }

    if (bitwiseAndBool(material.flags, GPU_METALLIC_ROUGHNESS_MAPPED)) {
        // See https://github.com/KhronosGroup/glTF-Sample-Viewer/blob/main/source/Renderer/shaders/material_info.glsl
        // See https://stackoverflow.com/questions/61140427/opengl-glsl-extract-metalroughness-map-to-metal-map-and-roughness-map
        vec2 metallicRoughness = texture(material.metallicRoughnessMap, texCoords).bg;
        //metallicRoughness *= vec2(metallic, roughness);
        metallic = metallicRoughness.x; //clamp(metallicRoughness.r / 2.0, 0.0, 1.0);
        roughness = metallicRoughness.y;
    }
    else {
        if (bitwiseAndBool(material.flags, GPU_ROUGHNESS_MAPPED)) {
            roughness = texture(material.roughnessMap, texCoords).r;
        }

        if (bitwiseAndBool(material.flags, GPU_METALLIC_MAPPED)) {
            metallic = texture(material.metallicMap, texCoords).r;
        }
    }

    if (bitwiseAndBool(material.flags, GPU_AMBIENT_MAPPED)) {
        ao = texture(material.ambientMap, texCoords).r;
    }

    // Coordinate space is set to world
    gPosition = fsPosition;
    // gNormal = (normal + 1.0) * 0.5; // Converts back to [-1, 1]
    gNormal = normal;
    gAlbedo = baseColor.rgb;
    gBaseReflectivity = material.baseReflectivity.xyz;
    gRoughnessMetallicAmbient = vec3(roughness, metallic, ao);
    //gStructureBuffer = calculateStructureOutput(fsViewSpacePos.z);
    gStructureBuffer = calculateStructureOutput(1.0 / gl_FragCoord.w);

    // Small offset to help prevent z fighting in certain cases
    if (baseColor.a < 1.0) {
        gl_FragDepth = gl_FragCoord.z - 0.00001;
    }
    else {
        gl_FragDepth = gl_FragCoord.z;
    }
}