STRATUS_GLSL_VERSION

in vec4 fsPosition;

uniform vec3 lightPos;
uniform float farPlane;

void main() {
    // get distance between fragment and light source
    float lightDistance = length(fsPosition.xyz - lightPos);
    
    // map to [0;1] range by dividing by far_plane
    lightDistance = lightDistance / farPlane;
    
    // write this as modified depth
    gl_FragDepth = lightDistance;
}  