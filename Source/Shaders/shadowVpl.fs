STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

#include "common.glsl"
#include "alpha_test.glsl"

smooth in vec4 fsPosition;
smooth in vec2 fsTexCoords;
flat in int fsDrawID;

uniform vec3 lightPos;
uniform float farPlane;

out vec3 color;

void main() {
    Material material = materials[materialIndices[fsDrawID]];
    vec4 baseColor = bitwiseAndBool(material.flags, GPU_DIFFUSE_MAPPED) ? texture(material.diffuseMap, fsTexCoords) : FLOAT4_TO_VEC4(material.diffuseColor);

    runAlphaTest(baseColor.a);

    //vec3 emissive = bool(fsEmissiveMapped) ? emissiveTextureMultiplier * texture(material.emissiveMap, texCoords).rgb : FLOAT3_TO_VEC3(material.emissiveColor);

    // get distance between fragment and light source
    float lightDistance = length(fsPosition.xyz - lightPos);
    
    // map to [0;1] range by dividing by far_plane
    lightDistance = saturate(lightDistance / farPlane);
    
    // write this as modified depth
    gl_FragDepth = lightDistance;

    color = baseColor.rgb;
}  