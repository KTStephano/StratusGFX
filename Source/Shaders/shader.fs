STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require
#extension GL_ARB_sparse_texture2 : require

out vec4 color;

void main() {
    color = vec4(1.0, 0.0, 1.0, 1.0);
}