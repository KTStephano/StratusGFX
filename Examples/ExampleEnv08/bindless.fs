STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

// SSBO containing the textures
layout(binding = 2, std430) readonly buffer ssbo3 {
    sampler2D textures[];
};

smooth in vec2 fsUv;
flat in vec3 fsNormal;
flat in int fsInstance;

out vec4 color;

void main() {
    sampler2D tex = textures[fsInstance];
    color = vec4(texture(tex, fsUv).rgb, 1.0);
    //color = vec4(1.0, 0.0, 0.0, 1.0);
}