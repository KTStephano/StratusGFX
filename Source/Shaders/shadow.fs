STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

#include "common.glsl"

smooth in vec4 fsPosition;
smooth in vec2 fsTexCoords;
flat in int fsDrawID;

uniform vec3 lightPos;
uniform float farPlane;

void main() {
    // get distance between fragment and light source
    float lightDistance = length(fsPosition.xyz - lightPos);
    
    // map to [0;1] range by dividing by far_plane
    lightDistance = saturate(lightDistance / farPlane);
    
    // write this as modified depth
    gl_FragDepth = lightDistance;
}  