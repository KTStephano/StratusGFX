#version 150 core

smooth in vec2 fsTexCoords;

//uniform vec3 diffuseColor;
uniform sampler2D diffuseTexture;
//uniform float gamma = 2.2;

out vec4 color;

void main() {
    vec3 texColor = texture(diffuseTexture, fsTexCoords).xyz;
    // Apply gamma correction
    //texColor = pow(texColor, vec3(1.0 / gamma));
    color = vec4(texColor, 1.0);
}