STRATUS_GLSL_VERSION

layout (location = 0)  in vec3 position;

out vec3 fsTexCoords;

uniform mat4 projection;
uniform mat4 view;

void main() {
    fsTexCoords = position;
    vec4 pos = projection * view * vec4(position, 1.0);
    // Set the z and w components to w so that w / w = 1.0 which
    // is the maximum depth value. This will allow the graphics pipeline
    // to know that we always want the skybox to fail the depth test if a valid
    // pixel is already in the buffer from a previous stage.
    gl_Position = pos.xyww;
}