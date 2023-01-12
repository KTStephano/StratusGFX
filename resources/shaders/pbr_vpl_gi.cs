STRATUS_GLSL_VERSION

layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

// GBuffer information
layout(binding = 0) readonly uniform image2D sampler2D gPosition;
layout(binding = 1) readonly uniform image2D sampler2D gNormal;
layout(binding = 2) readonly uniform image2D sampler2D gAlbedo;
layout(binding = 3) readonly uniform image2D sampler2D gBaseReflectivity;
layout(binding = 4) readonly uniform image2D sampler2D gRoughnessMetallicAmbient;

// Camera information
uniform vec3 viewPosition;

// in/out frame texture
layout (binding = 5) uniform image2D screen;

// Resident shadow maps
layout (std430, binding = 6) buffer vplShadowMaps {
    samplerCube shadowCubeMaps[];
};

void main() {
    vec2 pixelCoords = gl_GlobalInvocationID.xy;
    // We set all the local sizes to 1 so gl_NumWorkGroups.xy contains
    // screen width/height
    vec2 texCoords = pixelCoords / gl_NumWorkGroups.xy;
    // gl_NumWorkGroups.xy contains the screen width/height but
    // gl_NumWorkGroups.z contains one invocation per light
    int lightIndex = int(gl_GlobalInvocationID.z);

    vec3 baseColor = texture(gAlbedo, texCoords).rgb;
    vec3 normal = normalize(texture(gNormal, texCoords).rgb * 2.0 - vec3(1.0));
    float roughness = texture(gRoughnessMetallicAmbient, texCoords).r;
    float metallic = texture(gRoughnessMetallicAmbient, texCoords).g;
    // Note that we take the AO that may have been packed into a texture and augment it by SSAO
    // Note that singe SSAO is sampler2DRect, we need to sample in pixel coordinates and not texel coordinates
    float ambient = texture(gRoughnessMetallicAmbient, texCoords).b * texture(ssao, texCoords * vec2(windowWidth, windowHeight)).r;
    vec3 baseReflectivity = texture(gBaseReflectivity, texCoords).rgb;

    
}