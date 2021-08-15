#version 410 core

// Note that with length 5, we end up having a kernel size of 9x9 since the center counts as 1, then we go
// +4 in one direction, -4 in other direction while blurring
//
// Also note that with the type of Gaussian Blur we're using, we split a 9x9 blur kernel into two separate iterations
// of length 9. The nice property of Gaussian Blur is that the result is effectively the same as doing it as nested 9x9
// loops, but for way less computation.
#define WEIGHT_LENGTH 5

// See https://learnopengl.com/Advanced-Lighting/Bloom
uniform sampler2D image;
uniform bool horizontal;
// Notice that the weights decrease, which signifies the start (weight[0]) contributing most
// and last (weight[4]) contributing least
uniform float weights[WEIGHT_LENGTH] = float[] (0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

in vec2 fsTexCoords;

layout (location = 0) out vec4 fsColor;

void main() {
    vec3 color = texture(image, fsTexCoords).rgb * weights[0];
    // This will give us the size of a single texel in (x, y) directions
    // (the 0 is telling it to give us the size at mipmap 0, aka full size image)
    vec2 texelWidth = 1.0 / textureSize(image, 0);
    if (horizontal) {
        for (int i = 1; i < WEIGHT_LENGTH; ++i) {
            vec2 texOffset = vec2(texelWidth.x * i, 0.0);
            // Notice we to +- texOffset so we can calculate both directions at once from the starting pixel
            color = color + texture(image, fsTexCoords + texOffset).rgb * weights[i];
            color = color + texture(image, fsTexCoords - texOffset).rgb * weights[i];
        }
    }
    else {
        for (int i = 1; i < WEIGHT_LENGTH; ++i) {
            vec2 texOffset = vec2(0.0, texelWidth.y * i);
            color = color + texture(image, fsTexCoords + texOffset).rgb * weights[i];
            color = color + texture(image, fsTexCoords - texOffset).rgb * weights[i];
        }
    }

    fsColor = vec4(color, 1.0);
}