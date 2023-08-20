STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require
#extension GL_ARB_sparse_texture2 : require

in vec2 fsTexCoords;

uniform uint numPageGroupsX;
uniform uint numPageGroupsY;

layout (std430, binding = 1) readonly buffer inputBlock1 {
    uint pageGroupsToRender[];
};

out vec4 color;

void main() {
    uvec2 pageGroupCoords = uvec2(fsTexCoords * (vec2(numPageGroupsX, numPageGroupsY) - vec2(1.0)));
    uint pageGroupIndex = pageGroupCoords.x + pageGroupCoords.y * numPageGroupsX;

    uint value = pageGroupsToRender[pageGroupIndex];

    //color = vec4((value > 0 ? vec3(129.0 / 255.0, 1.0, 104.0 / 255.0) : vec3(1.0, 97.0 / 255.0, 97.0 / 255.0)), 1.0);
    color = vec4((value > 0 ? vec3(129.0 / 255.0, 1.0, 104.0 / 255.0) : vec3(0.0)), 1.0);
}