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
#include "atmospheric_postfx.glsl"
#include "pbr.glsl"

uniform sampler2DRect atmosphereBuffer;
uniform vec3 atmosphericLightPos;

#define MAX_LIGHTS 200
// Apple limits us to 16 total samplers active in the pipeline :(
#define MAX_SHADOW_LIGHTS 48
#define SPECULAR_MULTIPLIER 128.0
//#define AMBIENT_INTENSITY 0.00025

uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform sampler2D gAlbedo;
uniform sampler2D gBaseReflectivity;
uniform sampler2D gRoughnessMetallicAmbient;
uniform sampler2DRect ssao;

uniform float windowWidth;
uniform float windowHeight;

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
// If true then the light will be invisible when the sun is not overhead - 
// useful for brightening up directly-lit scenes without Static or RT GI
uniform bool lightBrightensWithSun[MAX_SHADOW_LIGHTS];
//uniform bool lightIsLightProbe[MAX_HAD]
// Since max lights is an upper bound, this can
// tell us how many lights are actually present
uniform int numLights = 0;
uniform int numShadowLights = 0;

/**
 * Information about the directional infinite light (if there is one)
 */
uniform bool infiniteLightingEnabled = false;
uniform vec3 infiniteLightColor;
// uniform float cascadeSplits[4];
// Allows us to take the texture coordinates and convert them to light space texture coordinates for cascade 0
// uniform mat4 cascade0ProjView;

in vec2 fsTexCoords;

layout (location = 0) out vec3 fsColor;

void main() {
    vec2 texCoords = fsTexCoords;
    vec3 fragPos = texture(gPosition, texCoords).rgb;
    vec3 viewDir = normalize(viewPosition - fragPos);

    vec3 baseColor = texture(gAlbedo, texCoords).rgb;
    vec3 normal = normalize(texture(gNormal, texCoords).rgb * 2.0 - vec3(1.0));
    float roughness = texture(gRoughnessMetallicAmbient, texCoords).r;
    float metallic = texture(gRoughnessMetallicAmbient, texCoords).g;
    // Note that we take the AO that may have been packed into a texture and augment it by SSAO
    // Note that singe SSAO is sampler2DRect, we need to sample in pixel coordinates and not texel coordinates
    float ambient = texture(gRoughnessMetallicAmbient, texCoords).b * texture(ssao, texCoords * vec2(windowWidth, windowHeight)).r;
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
                shadowFactor = calculateShadowValue(shadowCubeMaps[shadowCubeMapIndex], lightFarPlanes[shadowCubeMapIndex], fragPos, lightPositions[i], dot(lightPositions[i] - fragPos, normal), 27);
                // If true then the light will be invisible when the sun is not overhead - 
                // useful for brightening up directly-lit scenes without Static or RT GI
                if (lightBrightensWithSun[i] && infiniteLightingEnabled) {
                    vec3 cascadeBlends = vec3(dot(cascadePlanes[0], vec4(lightPositions[i], 1.0)),
                                      dot(cascadePlanes[1], vec4(lightPositions[i], 1.0)),
                                      dot(cascadePlanes[2], vec4(lightPositions[i], 1.0)));
                    shadowFactor = max(shadowFactor, 1.0 - calculateInfiniteShadowValue(vec4(lightPositions[i], 1.0), cascadeBlends, infiniteLightDirection));
                }
            }
            color = color + calculatePointLighting(fragPos, baseColor, normal, viewDir, lightPositions[i], lightColors[i], roughness, metallic, ambient, shadowFactor, baseReflectivity);
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
        color = color + calculateLighting(infiniteLightColor, lightDir, viewDir, normal, baseColor, roughness, metallic, ambient, shadowFactor, baseReflectivity, 1.0, worldLightAmbientIntensity);
    }
    else {
        color = color + baseColor * ambient * ambientIntensity;
    }

    fsColor = boundHDR(color);
}