#version 150 core

in vec2 fsTexCoords;

uniform vec3 diffuseColor;
uniform sampler2D diffuseTexture;

out vec4 color;

void main() {
    vec3 texColor = texture(diffuseTexture, fsTexCoords).xyz;
    color = vec4(texColor * diffuseColor, 1.0);
}