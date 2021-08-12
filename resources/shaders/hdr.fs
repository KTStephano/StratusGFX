#version 410 core

in vec2 fsTexCoords;

uniform sampler2D screen;
uniform float gamma = 2.2;

out vec4 color;

// This uses Reinhard tone mapping without exposure. More
// advanced techniques can be used to achieve a very different
// look and feel to the final output.
void main() {
    vec3 screenColor = texture(screen, fsTexCoords).rgb;

    vec3 reinhard = screenColor / (screenColor + vec3(1.0));
    vec3 corrected = pow(reinhard, vec3(1.0 / gamma));

    color = vec4(corrected, 1.0);
}