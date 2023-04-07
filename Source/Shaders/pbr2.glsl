STRATUS_GLSL_VERSION

#include "pbr.glsl"
#include "fog.glsl"

// All of the information was taken from these two sources:
//      https://google.github.io/filament/Filament.html
//      https://learnopengl.com/PBR/Theory

// Standard model BSDF (Bidirectional Scattering Distribution Function) is composed of two
// components: a BRDF (Bidirectional Reflectance Distribution Function) and a
// BTDF (Bidirectional Transmittance Function)
//
// We will be ignoring the BTDF so that our model will only account for BRDF contribution.
//
// The BRDF itself (both single scattering and multiscattering) is composed of two components:
//      Diffuse  Fd
//      Specular Fr
//
// Fd results from light bouncing around just below the surface before being scattered in many
// different directions. Fr only interacts with top level microfacets and is redirected in a much
// more concentrated direction resulting in bright specular reflections.
//
// Complete equation is as follows:
//      F(v, l) = Fd(v, l) + Fr(v, l)
// where v is the view unit vector and l is the incident light unit vector.
//
// At the core of this model is microfacet theory which states that surfaces, at the micro level,
// are not uniformly flat and instead are made up of very small bumps. For rendering, any microfacet
// whose normal is orientated halfway betwee the incoming light direction and view direction will show
// up as visible light. This is because its light is being redirected directly towards the camera.
//
// Components:
//      D (also NDF - Normal Distribution Function) models the distribution of the microfacets of the surface
//      G models visibility (occlusion) of microfacets, which is how much light is occluded by other nearby microfacets
//      F (Fresnel) models the fact that the amount of light the viewer sees reflected from a surface depends on the viewing angle
//        - this is known as the Fresnel Effect
//
// Dielectrics (non-metals) vs Conductors (metals)
//      A dielectric exhibits subsurface scattering which means it will have a diffuse and specular contribution
//      A pure conductor exhibits no subsurface scattering which means it only has a specular contribution
//
// Energy Conservation
//      Another important aspect of the standard PBR model states that the total amount of specular and diffuse reflectance
//      energy is less than the total amount of incident energy.

// The GGX NDF is a distribution with short peak highlights and long-tailed falloff
float NDF_GGX(float NdotH, float roughness) {
    float r2 = roughness * roughness;
    float f = (NdotH * r2 - NdotH) * NdotH + 1.0;
    return r2 / max(PI * f * f, PREVENT_DIV_BY_ZERO);
}

// This uses a height-correlated Smith function for determining geometric shadowing. What this means
// is that the height of the microfacets is taken into account to determine masking and shadowing.
float Visibility_G_SmithGGX(float NdotV, float NdotL, float roughness) {
    float a2 = roughness * roughness;
    float ggxv = NdotL * sqrt((NdotV - a2 * NdotV) * NdotV + a2);
    float ggxl = NdotV * sqrt((NdotL - a2 * NdotL) * NdotL + a2);
    return 0.5 / max(ggxv + ggxl, PREVENT_DIV_BY_ZERO);
}

// Fresnel Effect defines how light reflects and refracts at the interface between two different media/materials
// (such as a rock meeting the surface of water).
//
// Normally this function takes both a base reflectance at normal incidence (f0) and a reflectance at grazing angles
// (f90). For faster calculations but less accurace results, f90 can be set to 1.0.
//
// This interpolates between f and f0.
vec3 Fr_Fresnel_Schlick(float u, vec3 f0) {
    float f = pow(1.0 - u, 5.0);
    return f + f0 * (1.0 - f);
}

// Fresnel Effect defines how light reflects and refracts at the interface between two different media/materials
// (such as a rock meeting the surface of water).
//
// Normally this function takes both a base reflectance at normal incidence (f0) and a reflectance at grazing angles
// (f90). For faster calculations but less accurace results, f90 can be set to 1.0.
//
// This interpolates between f0 and f.
float Fd_Fresnel_Schlick(float u, float f0, float f90) {
    float f = pow(1.0 - u, 5.0);
    return f0 + (f90 - f0) * f;
}

// This is the Disney diffuse model - its final value is meant to be multiplied by diffuse color
float Fd_Burley(float NdotV, float NdotL, float LdotH, float roughness) {
    float f90 = 0.5 + 2.0 * roughness * LdotH * LdotH;
    float lightScatter = Fd_Fresnel_Schlick(NdotL, 1.0, f90);
    float viewScatter = Fd_Fresnel_Schlick(NdotV, 1.0, f90);
    return lightScatter * viewScatter * (1.0 / PI);
}

float Fd_Lambert(float NdotV, float NdotL, float LdotH, float roughness) {
    return 1.0 / PI;
}

vec3 singleScatteringBRDF_Specular(
    float NdotV,
    float NdotL,
    float NdotH,
    float LdotH,
    float remappedRoughness,
    vec3 f0) {

    float NDF        = NDF_GGX(NdotH, remappedRoughness);
    vec3  Fresnel    = Fr_Fresnel_Schlick(LdotH, f0);
    float Visibility = Visibility_G_SmithGGX(NdotV, NdotL, remappedRoughness);

    return (NDF * Visibility) * Fresnel;
}

vec3 singleScatteringBRDF_Diffuse(
    float NdotV,
    float NdotL,
    float NdotH,
    float LdotH,
    float remappedRoughness,
    vec3 diffuseColor) {

    return diffuseColor * Fd_Burley(NdotV, NdotL, LdotH, remappedRoughness);
}

// This uses single scattering which means it does not account for the fact that light may bounce around
// several microfacets and then still escape. A downside is that this tends to exhibit energy loss as roughness
// increases. This can be prevented by switching to a more accurace multi-scattering model (Filament paper explains
// how).
vec3 BRDF(
    vec3 lightDir, 
    vec3 viewDir, 
    vec3 normal, 
    vec3 baseColor, 
    vec3 baseReflectance, 
    float roughness, 
    float metallic) {
    
    // Remaps from perceptually linear roughness to roughness
    float remappedRoughness = roughness * roughness;

    // Compute diffuse from base using metallic value
    vec3 diffuseColor = (1.0 - metallic) * baseColor;

    // Compute reflectance - for purely metallic materials this is used as the diffuse color
    vec3 f0 = 0.16 * baseReflectance * baseReflectance * (1.0 - metallic) + baseColor * metallic;

    vec3 V = viewDir;
    vec3 L = normalize(lightDir);
    vec3 H = normalize(V + L);
    vec3 N = normal;

    float NdotV = saturate(dot(N, V)); //abs(dot(N, V)) + PREVENT_DIV_BY_ZERO;
    float NdotL = saturate(dot(N, L));
    float NdotH = saturate(dot(N, H));
    float LdotH = saturate(dot(L, H));

    // Specular
    vec3 Fr = singleScatteringBRDF_Specular(NdotV, NdotL, NdotH, LdotH, remappedRoughness, f0);

    // Diffuse
    vec3 Fd = singleScatteringBRDF_Diffuse(NdotV, NdotL, NdotH, LdotH, remappedRoughness, diffuseColor);

    // Does not account for light color/intensity (functions below do that)
    return (Fd + Fr) * NdotL;
}

vec3 BRDF_DiffuseOnly(
    vec3 lightDir, 
    vec3 viewDir, 
    vec3 normal, 
    vec3 baseColor, 
    vec3 baseReflectance, 
    float roughness, 
    float metallic) {
    
    // Remaps from perceptually linear roughness to roughness
    float remappedRoughness = roughness * roughness;

    // Compute diffuse from base using metallic value
    vec3 diffuseColor = (1.0 - metallic) * baseColor;

    // Compute reflectance - for purely metallic materials this is used as the diffuse color
    vec3 f0 = 0.16 * baseReflectance * baseReflectance * (1.0 - metallic) + baseColor * metallic;

    vec3 V = viewDir;
    vec3 L = normalize(lightDir);
    vec3 H = normalize(V + L);
    vec3 N = normal;

    float NdotV = saturate(dot(N, V)); //abs(dot(N, V)) + PREVENT_DIV_BY_ZERO;
    float NdotL = saturate(dot(N, L));
    float NdotH = saturate(dot(N, H));
    float LdotH = saturate(dot(L, H));

    // Diffuse
    vec3 Fd = singleScatteringBRDF_Diffuse(NdotV, NdotL, NdotH, LdotH, remappedRoughness, diffuseColor);

    // Does not account for light color/intensity (functions below do that)
    return Fd * NdotL;
}

vec3 calculateLighting2(
    vec3 lightColor, 
    vec3 lightDir, 
    vec3 viewDir, 
    vec3 normal, 
    vec3 baseColor,
    float viewDist, 
    float fogIntensity,
    float roughness, 
    float metallic, 
    float ambientOcclusion, 
    float shadowFactor, 
    vec3 baseReflectance, 
    float attenuationFactor, 
    float ambientIntensity) {

    vec3 brdf = BRDF(lightDir, viewDir, normal, baseColor, baseReflectance, roughness, metallic);

    vec3 ambient = brdf * ambientOcclusion * lightColor * ambientIntensity;
    vec3 finalBrightness = brdf * lightColor;

    return attenuationFactor * (ambient + shadowFactor * applyFog(finalBrightness, viewDist, fogIntensity));
}

vec3 calculateLighting_DiffuseOnly(
    vec3 lightColor, 
    vec3 lightDir, 
    vec3 viewDir, 
    vec3 normal, 
    vec3 baseColor, 
    float viewDist,
    float roughness, 
    float metallic, 
    float ambientOcclusion, 
    float shadowFactor, 
    vec3 baseReflectance, 
    float attenuationFactor, 
    float ambientIntensity) {

    vec3 brdf = BRDF_DiffuseOnly(lightDir, viewDir, normal, baseColor, baseReflectance, roughness, metallic);

    vec3 ambient = brdf * ambientOcclusion * lightColor * ambientIntensity;
    vec3 finalBrightnes = brdf * lightColor;

    return attenuationFactor * (ambient + shadowFactor * finalBrightnes);
}

vec3 calculateDirectionalLighting(
    vec3 lightColor,
    vec3 lightDir, 
    vec3 viewDir, 
    vec3 normal, 
    vec3 baseColor, 
    float viewDist,
    float roughness, 
    float metallic, 
    float ambientOcclusion, 
    float shadowFactor, 
    vec3 baseReflectance, 
    float ambientIntensity) {

    return calculateLighting2(lightColor, lightDir, viewDir, normal, baseColor, viewDist, 1.0, roughness, metallic, ambientOcclusion, 1.0 - shadowFactor, baseReflectance, 1.0, ambientIntensity);
}

vec3 calculatePointLighting2(
    vec3 fragPosition, 
    vec3 baseColor, 
    vec3 normal, 
    vec3 viewDir, 
    vec3 lightPos, 
    vec3 lightColor, 
    float viewDist,
    float roughness, 
    float metallic, 
    float ambientOcclusion, 
    float shadowFactor, 
    vec3 baseReflectance) {

    vec3 lightDir   = lightPos - fragPosition;
    //lightColor = vec3(277 / 255, 66 / 255, 52 / 255) * 800;

    return calculateLighting2(lightColor, lightDir, viewDir, normal, baseColor, viewDist, length(lightColor) / 12, roughness, metallic, ambientOcclusion, 1.0 - shadowFactor, baseReflectance, quadraticAttenuation(lightDir), pointLightAmbientIntensity);
}

vec3 calculateVirtualPointLighting2(
    vec3 fragPosition, 
    vec3 baseColor, 
    vec3 normal, 
    vec3 viewDir, 
    vec3 lightPos, 
    vec3 lightColor,
    float viewDist,
    float lightRadius, 
    float roughness, 
    float metallic, 
    float ambientOcclusion, 
    float shadowFactor, 
    vec3 baseReflectance) {

    vec3 lightDir   = lightPos - fragPosition;
    float adjustedShadowFactor = 1.0 - shadowFactor;
    adjustedShadowFactor = max(adjustedShadowFactor, 0.1);

    //return calculateLighting_DiffuseOnly(lightColor, lightDir, viewDir, normal, baseColor, viewDist, roughness, metallic, ambientOcclusion, adjustedShadowFactor, baseReflectance, vplAttenuation(lightDir, lightRadius), 0.0);
    return calculateLighting2(lightColor, lightDir, viewDir, normal, baseColor, viewDist, 0.0, roughness, metallic, ambientOcclusion, adjustedShadowFactor, baseReflectance, vplAttenuation(lightDir, lightRadius), 0.0);
}