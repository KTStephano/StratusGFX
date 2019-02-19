#version 150 core

in vec3 position;
in vec3 normal;
//in vec2 texCoords;

uniform mat4 projection;
// We need to do lighting calculations in either
// world space or eye space, so in this case we
// are going to choose eye space
uniform mat4 modelView;

out vec3 fsPosition;
out vec3 fsNormal;
//out vec2 fsTexCoords;

// See https://paroj.github.io/gltut/Illumination/Tut09%20Normal%20Transformation.html
// for information about inverse-transpose
void main() {
    mat3 modelViewNoTranslate = mat3(modelView);
    vec3 tempNormal = modelViewNoTranslate * normal;
    fsNormal = normalize(tempNormal);
    fsTexCoords = texCoords;
    vec4 pos = modelView * vec4(position, 1.0);
    fsPosition = pos.xyz;
    gl_Position = projection * pos;
}