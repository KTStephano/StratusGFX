STRATUS_GLSL_VERSION

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

#include "common.glsl"

uniform float infiniteLightZnear;
uniform float infiniteLightZfar;
uniform vec3 infiniteLightDirection;
uniform float infiniteLightDepthBias = 0.0;
uniform sampler2DArrayShadow infiniteLightShadowMap;
// Each vec4 offset has two pairs of two (x, y) texel offsets. For each cascade we sample
// a neighborhood of 4 texels and additive blend the results.
uniform vec4 shadowOffset[2];
// Represents a plane which transitions from 0 to 1 as soon as two cascades overlap
uniform vec4 cascadePlanes[3];
uniform mat4 cascadeProjViews[4];

uniform float worldLightAmbientIntensity = 0.003;
uniform float pointLightAmbientIntensity = 0.003;
uniform float ambientIntensity = 0.00025;

// Synchronized with definition found in StratusGpuCommon.h
#define MAX_TOTAL_SHADOW_ATLASES (14)
#define MAX_TOTAL_SHADOWS_PER_ATLAS (300)
#define MAX_TOTAL_SHADOW_MAPS (MAX_TOTAL_SHADOW_ATLASES * MAX_TOTAL_SHADOWS_PER_ATLAS)

// Synchronized with definition found in StratusGpuCommon.h
struct AtlasEntry {
    int index;
    int layer;
};

// Main idea came from https://learnopengl.com/Advanced-Lighting/Shadows/Point-Shadows
float calculateShadowValue8Samples(samplerCubeArray shadowMaps, int shadowIndex, float lightFarPlane, vec3 fragPos, vec3 lightPos, float lightNormalDotProduct, float minBias) {
    // Not required for fragDir to be normalized
    vec3 fragDir = fragPos - lightPos;
    float currentDepth = length(fragDir);

    // Part of this came from GPU Gems
    // @see http://developer.download.nvidia.com/books/HTML/gpugems/gpugems_ch12.html
    float bias = (currentDepth * max(0.5 * (1.0 - max(lightNormalDotProduct, 0.0)), minBias));// - texture(shadowCubeMap, fragDir).r;
    // Now we use a sampling-based method to look around the current pixel
    // and blend the values for softer shadows (introduces some blur). This falls
    // under the category of Percentage-Closer Filtering (PCF) algorithms.
    float shadow = 0.0;
    float offset = 0.3;
    float offsets[2] = float[](-offset, offset);
    // This should result in 2*2*2 = 8 samples
    for (int x = 0; x < 2; ++x) {
        for (int y = 0; y < 2; ++y) {
            for (int z = 0; z < 2; ++z) {
                float depth = textureLod(shadowMaps, vec4(fragDir + vec3(offsets[x], offsets[y], offsets[z]), float(shadowIndex)), 0).r;
                // It's very important to multiply by lightFarPlane. The recorded depth
                // is on the range [0, 1] so we need to convert it back to what it was originally
                // or else our depth comparison will fail.
                depth = depth * lightFarPlane;
                if ((currentDepth - bias) > depth) {
                    shadow = shadow + 1.0;
                }
            }
        }
    }

    return shadow / 8.0;
}

// Main idea came from https://learnopengl.com/Advanced-Lighting/Shadows/Point-Shadows
float calculateShadowValue1Sample(samplerCubeArray shadowMaps, int shadowIndex, float lightFarPlane, vec3 fragPos, vec3 lightPos, float lightNormalDotProduct, float minBias) {
    // Not required for fragDir to be normalized
    vec3 fragDir = fragPos - lightPos;
    float currentDepth = length(fragDir);

    // Part of this came from GPU Gems
    // @see http://developer.download.nvidia.com/books/HTML/gpugems/gpugems_ch12.html
    float bias = (currentDepth * max(0.5 * (1.0 - max(lightNormalDotProduct, 0.0)), minBias));// - texture(shadowCubeMap, fragDir).r;
    float shadow = 0.0;
    float depth = textureLod(shadowMaps, vec4(fragDir, float(shadowIndex)), 0).r;
    // It's very important to multiply by lightFarPlane. The recorded depth
    // is on the range [0, 1] so we need to convert it back to what it was originally
    // or else our depth comparison will fail.
    depth = depth * lightFarPlane;
    if ((currentDepth - bias) > depth) {
        shadow = 1.0;
    }

    return shadow;
}

// See https://developer.download.nvidia.com/books/HTML/gpugems/gpugems_ch11.html
float sampleShadowTexture(sampler2DArrayShadow shadow, vec4 coords, float depth, vec2 offset, float bias) {
    coords.w = depth - bias;
    coords.xy += offset;
    return textureLod(shadow, coords, 0);
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
float calculateInfiniteShadowValue(vec4 fragPos, vec3 cascadeBlends, vec3 normal, bool useDepthBias) {
	// Since dot(l, n) = cos(theta) when both are normalized, below should compute tan theta
    // See: http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-16-shadow-mapping/
	//float tanTheta = tan(acos(dot(normalize(infiniteLightDirection), normal)));
    //float bias = 0.005 * tanTheta;
    //bias = -clamp(bias, 0.0, 0.01);
    //float bias = 2e-19;
    float bias = infiniteLightDepthBias / (infiniteLightZfar - infiniteLightZnear);
    if (!useDepthBias) {
        bias = 0.0;
    }

    vec4 position = fragPos;
    position.xyz += normal * ( 1.0f - saturate( dot( normal, infiniteLightDirection ) ) ) * 1.0;

    vec4 p1, p2;
    vec3 cascadeCoords[4];
    // cascadeCoords[0] = cascadeCoord0 * 0.5 + 0.5;
    for (int i = 0; i < 4; ++i) {
        // cascadeProjViews[i] * position puts the coordinates into clip space which are on the range of [-1, 1].
        // Since we are looking for texture coordinates on the range [0, 1], we first perform the perspective divide
        // and then perform * 0.5 + vec3(0.5).
        vec4 coords = cascadeProjViews[i] * position;
        cascadeCoords[i] = coords.xyz / coords.w; // Perspective divide
        cascadeCoords[i].xyz = cascadeCoords[i].xyz * 0.5 + vec3(0.5);
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

    vec2 wh = computeTexelSize(infiniteLightShadowMap, 0);
                         
    float light1 = 0.0;
    float light2 = 0.0;
    float samples = 0.0;
    p1.xy = shadowCoord1;
    p2.xy = shadowCoord2;
    // 16-sample filtering - see https://developer.download.nvidia.com/books/HTML/gpugems/gpugems_ch11.html
    float bound = 1.0; // 1.5 = 16 sample; 1.0 = 4 sample
    for (float y = -bound; y <= bound; y += 1.0) {
        for (float x = -bound; x <= bound; x += 1.0) {
            light1 += sampleShadowTexture(infiniteLightShadowMap, p1, depth1, vec2(x, y) * wh, bias);
            light2 += sampleShadowTexture(infiniteLightShadowMap, p2, depth2, vec2(x, y) * wh, bias);
            ++samples;
        }
    }

    // blend and return
    return mix(light2, light1, weight) * (1.0 / samples); //* 0.25;
}

float normalDistribution(float NdotH, float roughness) {
    float roughness2 = roughness * roughness;
    float roughness4 = roughness2 * roughness2;

    float denominator = (NdotH * NdotH) * (roughness4 - 1.0) + 1.0;
    denominator = PI * (denominator * denominator);
    return roughness4 / max(denominator, PREVENT_DIV_BY_ZERO);
}

vec3 fresnel(vec3 albedo, float HdotV, vec3 baseReflectivity, float metallic) {
    vec3 F0 = mix(baseReflectivity, albedo, metallic);
    return F0 + (1.0 - F0) * pow(clamp(1.0 - HdotV, 0.0, 1.0), 5);
}

float geometrySchlickGGX(float NdotX, float k) {
    return NdotX / max(NdotX * (1 - k) + k, PREVENT_DIV_BY_ZERO);
}

float geometry(vec3 normal, vec3 viewDir, vec3 lightDir, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    float NdotV = max(dot(normal, viewDir), 0.0);
    float NdotL = max(dot(normal, lightDir), 0.0);
    return geometrySchlickGGX(NdotV, k) * geometrySchlickGGX(NdotL, k);
}

float quadraticAttenuation(vec3 lightDir) {
    float lightDist = length(lightDir);
    return 1.0 / (1.0 + lightDist * lightDist);
}

float vplAttenuation(vec3 lightDir, float lightRadius) {
    float minDist = 10.0 * lightRadius;
    float maxDist = 0.75 * lightRadius;
    //float lightDist = max(length(lightDir), minDist);
    float lightDist = length(lightDir);
    //float lightDist = clamp(length(lightDir), minDist, maxDist);
    return 1.0 / (minDist + 1.0 * lightDist * lightDist);
}

vec3 calculateLighting(vec3 lightColor, vec3 lightDir, vec3 viewDir, vec3 normal, vec3 baseColor, 
    float roughness, float metallic, float ao, float shadowFactor, vec3 baseReflectivity, 
    float attenuationFactor, float ambientIntensity) {
    
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
    vec3 kD        = (vec3(1.0) - kS) * (1.0 - metallic); // TODO: UNCOMMENT METALLIC PART
    float D        = normalDistribution(NdotH, roughness);
    float G        = geometry(N, V, L, roughness);
    vec3 radiance  = lightColor; // * attenuationFactor;
    vec3 specular  = (D * F * G) / max((4 * W0dotN * WidotN), PREVENT_DIV_BY_ZERO);

    //float atmosphericIntensity = getAtmosphericIntensity(atmosphereBuffer, lightColor, fsTexCoords * vec2(windowWidth, windowHeight));

    vec3 ambient = baseColor * ao * lightColor * ambientIntensity; // * attenuationFactor;
    vec3 finalBrightnes = (kD * baseColor / PI + specular) * radiance * NdotWi;

    //return (1.0 - shadowFactor) * ((kD * baseColor / PI + specular) * diffuse * NdotWi);
    return attenuationFactor * (ambient + shadowFactor * finalBrightnes);
}

vec3 calculateDiffuseOnlyLighting(vec3 lightColor, vec3 lightDir, vec3 viewDir, vec3 normal, vec3 baseColor, 
    float metallic, float ao, float shadowFactor, vec3 baseReflectivity, float attenuationFactor, float ambientIntensity) {
    
    vec3 V = viewDir;
    vec3 L = normalize(lightDir);
    vec3 H = normalize(V + L);
    vec3 N = normal;

    float NdotH    = max(dot(N, H), 0.0);
    float HdotV    = max(dot(H, V), 0.0);
    float NdotWi   = max(dot(N, L), 0.0);
    vec3 F         = fresnel(baseColor, clamp(HdotV, 0.0, 1.0), baseReflectivity, metallic);
    vec3 kS        = F;
    // We multiply by inverse of metallic since we only want non-metals to have diffuse lighting
    vec3 kD        = (vec3(1.0) - kS) * (1.0 - metallic); // TODO: UNCOMMENT METALLIC PART
    vec3 radiance  = lightColor; // * attenuationFactor;

    //float atmosphericIntensity = getAtmosphericIntensity(atmosphereBuffer, lightColor, fsTexCoords * vec2(windowWidth, windowHeight));

    vec3 ambient = baseColor * ao * lightColor * ambientIntensity; // * attenuationFactor;
    vec3 finalBrightnes = (kD * baseColor / PI) * radiance * NdotWi;

    //return (1.0 - shadowFactor) * ((kD * baseColor / PI + specular) * diffuse * NdotWi);
    return attenuationFactor * (ambient + shadowFactor * finalBrightnes);
}

vec3 calculatePointLighting(vec3 fragPosition, vec3 baseColor, vec3 normal, vec3 viewDir, vec3 lightPos, vec3 lightColor, float roughness, float metallic, float ao, float shadowFactor, vec3 baseReflectivity) {
    vec3 lightDir   = lightPos - fragPosition;

    return calculateLighting(lightColor, lightDir, viewDir, normal, baseColor, roughness, metallic, ao, 1.0 - shadowFactor, baseReflectivity, quadraticAttenuation(lightDir), pointLightAmbientIntensity);
}

vec3 calculateVirtualPointLighting(vec3 fragPosition, vec3 baseColor, vec3 normal, vec3 viewDir, vec3 lightPos, vec3 lightColor,
    float lightRadius, float roughness, float metallic, float ao, float shadowFactor, vec3 baseReflectivity) {

    vec3 lightDir   = lightPos - fragPosition;

    //return calculateLighting(lightColor, lightDir, viewDir, normal, baseColor, roughness, metallic, ao, 1.0 - shadowFactor, baseReflectivity, vplAttenuation(lightDir, lightRadius), 0.0);
    return calculateDiffuseOnlyLighting(lightColor, lightDir, viewDir, normal, baseColor, metallic, ao, 1.0 - shadowFactor, baseReflectivity, vplAttenuation(lightDir, lightRadius), 0.003);
}