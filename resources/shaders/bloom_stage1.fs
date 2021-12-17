#version 400 core

in vec2 fsTexCoords;

uniform sampler2D image;

layout (location = 0) out vec4 fsBrightSpot;

void main() {
    vec4 color = vec4(texture(image, fsTexCoords).rgb, 1.0);
    // See https://learnopengl.com/Advanced-Lighting/Bloom
    vec3 weights = normalize(vec3(0.2126, 0.7152, 0.0722);
    float brightness = color.r * weights.r + color.g * weights.g + color.b * weights.b;
    if (brightness > 10.0) {
        fsBrightSpot = color;
    }
    else {
        fsBrightSpot = vec4(vec3(0.0), 1.0);
    }
}