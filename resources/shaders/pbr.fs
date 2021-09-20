#version 410 core

/**
 * @see https://learnopengl.com/PBR/Theory
 *
 * The equation we're working off of is Lo(p,ωo)=Integral [(kd(c/π)+DFG/(4(ωo⋅n)(ωi⋅n)))*Li(p,ωi)n⋅ωi] dωi
 *
 * We effectively discretize the function by just calculating it from the perspective of each light rather than
 * trying to solve the indefinite integral (which from what I understand has no analytical solution). This is also
 * known as a numerical solution where a finite number of steps approximate the real solution.
 *
 * Lo(p, wo) effectively stands for the light intensity (radiance) at point p when viewed from wo.
 *
 * kd is equal to 1.0 - ks, where ks is the specular component (roughly F in the above)
 *
 * (c/π) is equal to (surface color aka diffuse / pi), where the division by pi normalizes the color.
 *
 * DFG is three separate functions D, F, G:
 *      D = normal distribution function = α^2 / (π ((n⋅h) ^ 2 (α ^ 2 − 1)+1) ^ 2)
          ==> Approximates the total surface area of the microfacets which are perfectly aligned with the half angle vector
          ==> a is the surface's roughness
          ==> n is the surface normal
          ==> h is the half angle vector = normalize (l + v) = (l + v) / || l + v ||
          ==> Since n and h are normalized, n dot h is equal to cos(angle between them)
        F = Fresnel (Freh-nel) equation = F0+(1−F0)(1−(h⋅v)) ^ 5
          ==> Describes the ratio of the light that gets reflected (specular) over the light that gets refracted (diffuse)
          ==> A good approximation of specular component from earlier lighting approximations such as Blinn-Phong
          ==> F0 represents the base reflectivity of the surface calculated using indices of refraction (IOR)
              as the viewing angle gets closer and closer to 90 degrees, the stronger the Fresnel and thus reflections
              (demonstrated by looking at even a rough surface from a 90 degree angle - generally you will see some shine)
          ==> https://refractiveindex.info/ is a good place for F0
          ==> h*v is the dot product of normalized half angle vector and normalized view vector, so it is cos(angle between them)
        G = G(n,v,l,k)=Gsub(n,v,k)Gsub(n,l,k)
          ==> Approximates the amount of surface area where parts of the geometry occlude other parts of the surface (self-shadowing)
          ==> n is the surface normal
          ==> v is the view direction
          ==> k is a remapping of roughness based on whether we're using direct or IBL lighting
              (direct) = (a + 1) ^ 2 / 8
              (IBL)    = a ^ 2 / 2
          ==> l is the light direction vector
        Gsub = G_Schlick_GGX = n⋅v / ((n⋅v)(1−k)+k) (see above for definition of terms)

 * Li(p, wi) is the radiance of point p when viewed from light Wi. It is equal to (light_color * attenuation)
          ==> attenuation when using quadratic attenuation is equal to 1.0 / distance ^ 2
          ==> distance = length(light_pos - world_pos)

 * n*wi is equal to cos(angle between n and light wi) since both are normalized
 * wi = normalize(light_pos - world_pos)
 */

#define MAX_LIGHTS 200
// Apple limits us to 16 total samplers active in the pipeline :(
#define MAX_SHADOW_LIGHTS 11
#define SPECULAR_MULTIPLIER 128.0
#define WORLD_LIGHT_AMBIENT_INTENSITY 0.009
#define POINT_LIGHT_AMBIENT_INTENSITY 0.03
//#define AMBIENT_INTENSITY 0.00025
#define PI 3.14159265359
#define PREVENT_DIV_BY_ZERO 0.00001

uniform float ambientIntensity = 0.00025;

uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform sampler2D gAlbedo;
uniform sampler2D gBaseReflectivity;
uniform sampler2D gRoughnessMetallicAmbient;

/**
 * Information about the camera
 */
uniform vec3 viewPosition;

/**
 * Lighting information. All values related
 * to positions should be in world space.
 */
uniform vec3 lightPositions[MAX_LIGHTS];
uniform vec3 lightColors[MAX_LIGHTS];
uniform float lightRadii[MAX_LIGHTS];
uniform bool lightCastsShadows[MAX_LIGHTS];
uniform samplerCube shadowCubeMaps[MAX_SHADOW_LIGHTS];
uniform float lightFarPlanes[MAX_SHADOW_LIGHTS];
// Since max lights is an upper bound, this can
// tell us how many lights are actually present
uniform int numLights = 0;
uniform int numShadowLights = 0;

/**
 * Information about the directional infinite light (if there is one)
 */
uniform bool infiniteLightingEnabled = false;
uniform vec3 infiniteLightDirection;
uniform vec3 infiniteLightColor;
uniform sampler2DArrayShadow infiniteLightShadowMap;
// Each vec4 offset has two pairs of two (x, y) texel offsets. For each cascade we sample
// a neighborhood of 4 texels and additive blend the results.
uniform vec4 shadowOffset[2];
// cascadeScale and cascadeOffset represent the scale and translation components of a matrix
// which can convert from texture coordinates in cascade 0 to texture coordinates in any of the other
// cascades.
// uniform vec3 cascadeScale[3];
// uniform vec3 cascadeOffset[3];
// Represents a plane which transitions from 0 to 1 as soon as two cascades overlap
uniform vec4 cascadePlanes[3];
uniform mat4 cascadeProjViews[4];
// uniform float cascadeSplits[4];
// Allows us to take the texture coordinates and convert them to light space texture coordinates for cascade 0
// uniform mat4 cascade0ProjView;

in vec2 fsTexCoords;

layout (location = 0) out vec3 fsColor;

// Prevents HDR color values from exceeding 16-bit color buffer range
vec3 boundHDR(vec3 value) {
    return min(value, 65500.0);
}

// See https://community.khronos.org/t/saturate/53155
vec3 saturate(vec3 value) {
    return clamp(value, 0.0, 1.0);
}

float saturate(float value) {
    return clamp(value, 0.0, 1.0);
}

vec2 computeTexelWidth(sampler2DArrayShadow tex, int miplevel) {
    // This will give us the size of a single texel in (x, y) directions
    // (miplevel is telling it to give us the size at mipmap *miplevel*, where 0 would mean full size image)
    return (1.0 / textureSize(tex, miplevel).xy);// * vec2(2.0, 1.0);
}

#define PCF_SAMPLES 2.5;
#define PCF_SAMPLES_CUBED 16.0; // Not exactly... constrained it to 16 for Mac
#define PCF_SAMPLES_HALF 1;

float calculateShadowValue(vec3 fragPos, vec3 lightPos, int lightIndex, float lightNormalDotProduct) {
    // Not required for fragDir to be normalized
    vec3 fragDir = fragPos - lightPos;
    float currentDepth = length(fragDir);
    // It's very important to multiply by lightFarPlane. The recorded depth
    // is on the range [0, 1] so we need to convert it back to what it was originally
    // or else our depth comparison will fail.
    float calculatedDepth = texture(shadowCubeMaps[lightIndex], fragDir).r * lightFarPlanes[lightIndex];
    // This bias was found through trial and error... it was originally
    // 0.05 * (1.0 - max...)
    // Part of this came from GPU Gems
    // @see http://developer.download.nvidia.com/books/HTML/gpugems/gpugems_ch12.html
    float bias = (currentDepth * max(0.5 * (1.0 - max(lightNormalDotProduct, 0.0)), 0.05));// - texture(shadowCubeMap, fragDir).r;
    //float bias = max(0.75 * (1.0 - max(lightNormalDotProduct, 0.0)), 0.05);
    //float bias = max(0.85 * (1.0 - lightNormalDotProduct), 0.05);
    //bias = bias < 0.0 ? bias * -1.0 : bias;
    // Now we use a sampling-based method to look around the current pixel
    // and blend the values for softer shadows (introduces some blur). This falls
    // under the category of Percentage-Closer Filtering (PCF) algorithms.
    float shadow = 0.0;
    float samples = PCF_SAMPLES;
    float totalSamples = PCF_SAMPLES_CUBED; // 64 if samples is set to 4.0
    float offset = 0.1;
    float increment = offset / PCF_SAMPLES_HALF;
    for (float x = -offset; x < offset; x += increment) {
        for (float y = -offset; y < offset; y += increment) {
            for (float z = -offset; z < offset; z += increment) {
                float depth = texture(shadowCubeMaps[lightIndex], fragDir + vec3(x, y, z)).r;
                // Perform this operation to go from [0, 1] to
                // the original value
                depth = depth * lightFarPlanes[lightIndex];
                if ((currentDepth - bias) > depth) {
                    shadow = shadow + 1.0;
                }
            }
        }
    }

    //float bias = 0.005 * tan(acos(max(lightNormalDotProduct, 0.0)));
    //bias = clamp(bias, 0, 0.01);
    //return (currentDepth - bias) > calculatedDepth ? 1.0 : 0.0;
    return shadow / totalSamples;
}

// See https://developer.download.nvidia.com/books/HTML/gpugems/gpugems_ch11.html
float sampleShadowTexture(sampler2DArrayShadow shadow, vec4 coords, float depth, vec2 offset, float bias) {
    coords.w = depth - bias;
    coords.xy += offset;
    return texture(shadow, coords);
    // float closestDepth = texture(shadow, coords).r;
    // // 0.0 means not in shadow, 1.0 means fully in shadow
    // return depth > closestDepth ? 1.0 : 0.0;
    // return closestDepth;
}

// For more information, see:
//      "Foundations of Game Development, Volume 2: Rendering", pp. 189
//      https://learnopengl.com/Advanced-Lighting/Shadows/Shadow-Mapping
//      https://ogldev.org/www/tutorial49/tutorial49.html
//      https://alextardif.com/shadowmapping.html
//      https://johanmedestrom.wordpress.com/2016/03/18/opengl-cascaded-shadow-maps/
//      https://johanmedestrom.wordpress.com/2016/03/18/opengl-cascaded-shadow-maps/
//      https://developer.download.nvidia.com/books/HTML/gpugems/gpugems_ch11.html
//      http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-16-shadow-mapping/
float calculateInfiniteShadowValue(vec4 fragPos, vec3 cascadeBlends, vec3 normal) {
	// Since dot(l, n) = cos(theta) when both are normalized, below should compute tan theta
    // See: http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-16-shadow-mapping/
	float tanTheta = tan(acos(dot(normalize(infiniteLightDirection), normal)));
    float bias = 0.005 * tanTheta;
    bias = clamp(bias, 0.0, 0.01);

    vec4 p1, p2;
    vec3 cascadeCoords[4];
    // cascadeCoords[0] = cascadeCoord0 * 0.5 + 0.5;
    for (int i = 0; i < 4; ++i) {
        // cascadeCoords[i] = cascadeCoord0 * cascadeScale[i - 1] + cascadeOffset[i - 1];
        vec4 coords = cascadeProjViews[i] * fragPos;
        cascadeCoords[i] = coords.xyz / coords.w; // Perspective divide
        cascadeCoords[i].z = cascadeCoords[i].z * 0.5 + 0.5;
    }

    bool beyondCascade2 = cascadeBlends.y >= 0.0;
    bool beyondCascade3 = cascadeBlends.z >= 0.0;
    // p1.z = float(beyondCascade2) * 2.0;
    // p2.z = float(beyondCascade3) * 2.0 + 1.0;

    int index1 = beyondCascade2 ? 2 : 0;
    int index2 = beyondCascade3 ? 3 : 1;
    p1.z = float(index1);
    p2.z = float(index2);

    vec2 shadowCoord1 = cascadeCoords[index1].xy;
    vec2 shadowCoord2 = cascadeCoords[index2].xy;
    // Convert from range [-1, 1] to [0, 1]
    // shadowCoord1 = shadowCoord1 * 0.5 + 0.5;
    // shadowCoord2 = shadowCoord2 * 0.5 + 0.5;
    float depth1 = cascadeCoords[index1].z;
    float depth2 = cascadeCoords[index2].z;
    // Clamp depths between [0, 1] for final cascade to prevent darkening beyond bounds
    depth2 = beyondCascade3 ? saturate(depth2) : depth2;

    //vec3 blend = saturate(vec3(cascadeBlend[0], cascadeBlend[1], cascadeBlend[2]));
    float weight = beyondCascade2 ? saturate(cascadeBlends.y) - saturate(cascadeBlends.z) : 1.0 - saturate(cascadeBlends.x);

    vec2 wh = computeTexelWidth(infiniteLightShadowMap, 0);

    float light1 = 0.0;
    float samples = 0.0;
    p1.xy = shadowCoord1;
    // 16-sample filtering - see https://developer.download.nvidia.com/books/HTML/gpugems/gpugems_ch11.html
    for (float y = -1.5; y <= 1.5; y += 1.0) {
        for (float x = -1.5; x <= 1.5; x += 1.0) {
            light1 += sampleShadowTexture(infiniteLightShadowMap, p1, depth1, vec2(x, y) * wh, bias);
            ++samples;
        }
    }
    // Sample four times from first cascade
    // light1 += sampleShadowTexture(infiniteLightShadowMap, p1, depth1, shadowOffset[0].xy, bias);
    // light1 += sampleShadowTexture(infiniteLightShadowMap, p1, depth1, shadowOffset[0].zw, bias);
    // light1 += sampleShadowTexture(infiniteLightShadowMap, p1, depth1, shadowOffset[1].xy, bias);
    // light1 += sampleShadowTexture(infiniteLightShadowMap, p1, depth1, shadowOffset[1].zw, bias);


    float light2 = 0.0;
    // Sample four times from second cascade
    p2.xy = shadowCoord2;
    // 16-sample filtering
    for (float y = -1.5; y <= 1.5; y += 1.0) {
        for (float x = -1.5; x <= 1.5; x += 1.0) {
            light2 += sampleShadowTexture(infiniteLightShadowMap, p2, depth2, vec2(x, y) * wh, bias);
        }
    }
    // Sample four times from second cascade
    // light2 += sampleShadowTexture(infiniteLightShadowMap, p2, depth2, shadowOffset[0].xy, bias);
    // light2 += sampleShadowTexture(infiniteLightShadowMap, p2, depth2, shadowOffset[0].zw, bias);
    // light2 += sampleShadowTexture(infiniteLightShadowMap, p2, depth2, shadowOffset[1].xy, bias);
    // light2 += sampleShadowTexture(infiniteLightShadowMap, p2, depth2, shadowOffset[1].zw, bias);

    // blend and return
    return mix(light2, light1, weight) * (1.0 / samples); //* 0.25;
}

float normalDistribution(const float NdotH, const float roughness) {
    float roughnessSquared = roughness * roughness;
    float denominator = (NdotH * NdotH) * (roughnessSquared - 1) + 1;
    denominator = PI * (denominator * denominator);
    return roughnessSquared / max(denominator, PREVENT_DIV_BY_ZERO);
}

vec3 fresnel(vec3 albedo, float HdotV, vec3 baseReflectivity, float metallic) {
    vec3 F0 = mix(baseReflectivity, albedo, metallic);
    return F0 + (1.0 - F0) * pow(1.0 - HdotV, 5);
}

float geometrySchlickGGX(float NdotX, float k) {
    return NdotX / max(NdotX * (1 - k) + k, PREVENT_DIV_BY_ZERO);
}

float geometry(vec3 normal, vec3 viewDir, vec3 lightDir, const float roughness) {
    float k = pow(roughness + 1, 2) / 8.0;
    float NdotV = max(dot(normal, viewDir), 0.0);
    float NdotL = max(dot(normal, lightDir), 0.0);
    return geometrySchlickGGX(NdotV, k) * geometrySchlickGGX(NdotL, k);
}

vec3 calculatePointAmbient(vec3 fragPosition, vec3 baseColor, int lightIndex, const float ao) {
    vec3 lightPos   = lightPositions[lightIndex];
    vec3 lightColor = lightColors[lightIndex];
    vec3 lightDir   = lightPos - fragPosition;
    float lightDist = length(lightDir);
    float attenuationFactor = 1.0 / (1.0 + lightDist * lightDist);
    vec3 ambient = baseColor * ao * lightColor * POINT_LIGHT_AMBIENT_INTENSITY;
    return attenuationFactor * ambient;
}

float quadraticAttenuation(vec3 lightDir) {
    float lightDist = length(lightDir);
    return 1.0 / (1.0 + lightDist * lightDist);
}

vec3 calculateLighting(vec3 lightColor, vec3 lightDir, vec3 viewDir, vec3 normal, vec3 baseColor, const float roughness, const float metallic, const float ao, const float shadowFactor, vec3 baseReflectivity, 
    const float attenuationFactor, const float ambientIntensity) {
    
    vec3 V = viewDir;
    vec3 L = normalize(lightDir);
    vec3 H = normalize(V + L);
    vec3 N = normal;

    float NdotH    = max(dot(N, H), 0.0);
    float HdotV    = max(dot(H, V), 0.0);
    float W0dotN   = max(dot(V, N), 0.0);
    float WidotN   = max(dot(L, N), 0.0);
    float NdotWi   = max(dot(N, L), 0.0);
    vec3 F         = fresnel(baseColor, clamp(HdotV, 0.0, 1.0), baseReflectivity, metallic);
    vec3 kS        = F;
    // We multiply by inverse of metallic since we only want non-metals to have diffuse lighting
    vec3 kD        = (vec3(1.0) - kS) * (1.0 - metallic);
    float D        = normalDistribution(NdotH, roughness);
    float G        = geometry(N, V, L, roughness);
    vec3 diffuse   = lightColor; // * attenuationFactor;
    vec3 specular  = (D * F * G) / max((4 * W0dotN * WidotN), PREVENT_DIV_BY_ZERO);

    vec3 ambient = baseColor * ao * lightColor * ambientIntensity; // * attenuationFactor;

    //return (1.0 - shadowFactor) * ((kD * baseColor / PI + specular) * diffuse * NdotWi);
    return attenuationFactor * (ambient + shadowFactor * ((kD * baseColor / PI + specular) * diffuse * NdotWi));  
}

vec3 calculatePointLighting(vec3 fragPosition, vec3 baseColor, vec3 normal, vec3 viewDir, int lightIndex, const float roughness, const float metallic, const float ao, const float shadowFactor, vec3 baseReflectivity) {
    vec3 lightPos   = lightPositions[lightIndex];
    vec3 lightColor = lightColors[lightIndex];
    vec3 lightDir   = lightPos - fragPosition;

    return calculateLighting(lightColor, lightDir, viewDir, normal, baseColor, roughness, metallic, ao, 1.0 - shadowFactor, baseReflectivity, quadraticAttenuation(lightDir), POINT_LIGHT_AMBIENT_INTENSITY);
}

void main() {
    vec2 texCoords = fsTexCoords;
    vec3 fragPos = texture(gPosition, texCoords).rgb;
    vec3 viewDir = normalize(viewPosition - fragPos);

    vec3 baseColor = texture(gAlbedo, texCoords).rgb;
    vec3 normal = texture(gNormal, texCoords).rgb;
    float roughness = texture(gRoughnessMetallicAmbient, texCoords).r;
    float metallic = texture(gRoughnessMetallicAmbient, texCoords).g;
    float ambient = texture(gRoughnessMetallicAmbient, texCoords).b;
    vec3 baseReflectivity = texture(gBaseReflectivity, texCoords).rgb;

    vec3 color = vec3(0.0);
    int shadowIndex = 0;
    for (int i = 0; i < numLights; ++i) {
        //if (i >= numLights) break;

        // Check if we should perform any sort of shadow calculation
        int shadowCubeMapIndex = MAX_LIGHTS;
        float shadowFactor = 0.0;
        if (shadowIndex < numShadowLights && lightCastsShadows[i] == true) {
            // The cube maps are indexed independently from the light index
            shadowCubeMapIndex = shadowIndex;
            ++shadowIndex;
        }

        // calculate distance between light source and current fragment
        float distance = length(lightPositions[i] - fragPos);
        if(distance < lightRadii[i]) {
            if (shadowCubeMapIndex < MAX_LIGHTS) {
                shadowFactor = calculateShadowValue(fragPos, lightPositions[i], shadowCubeMapIndex, dot(lightPositions[i] - fragPos, normal));
            }
            color = color + calculatePointLighting(fragPos, baseColor, normal, viewDir, i, roughness, metallic, ambient, shadowFactor, baseReflectivity);
        }
        else if (distance < 2 * lightRadii[i]) {
            color = color + calculatePointAmbient(fragPos, baseColor, i, ambient);
        }
    }

    if (infiniteLightingEnabled) {
        vec3 lightDir = infiniteLightDirection;
        // vec3 cascadeCoord0 = (cascade0ProjView * vec4(fragPos, 1.0)).rgb;
        // cascadeCoord0 = cascadeCoord0 * 0.5 + 0.5;
        vec3 cascadeBlends = vec3(dot(cascadePlanes[0], vec4(fragPos, 1.0)),
                                  dot(cascadePlanes[1], vec4(fragPos, 1.0)),
                                  dot(cascadePlanes[2], vec4(fragPos, 1.0)));
        float shadowFactor = calculateInfiniteShadowValue(vec4(fragPos, 1.0), cascadeBlends, normal);
        //vec3 lightDir = infiniteLightDirection;
        color = color + calculateLighting(infiniteLightColor, lightDir, viewDir, normal, baseColor, roughness, metallic, ambient, shadowFactor, baseReflectivity, 1.0, WORLD_LIGHT_AMBIENT_INTENSITY);
    }
    else {
        color = color + baseColor * ambient * ambientIntensity;
    }

    fsColor = boundHDR(color);
}