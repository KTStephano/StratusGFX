STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

#include "mesh_data.glsl"

layout (std430, binding = 13) readonly buffer SSBO3 {
    mat4 modelMatrices[];
};

layout (std430, binding = 14) readonly buffer SSBO4 {
    mat4 prevModelMatrices[];
};

uniform mat4 projectionView;
uniform mat4 jitterProjectionView;
uniform mat4 prevProjectionView;

uniform int viewWidth;
uniform int viewHeight;

//uniform mat4 modelView;

smooth out vec2 fsTexCoords;
flat out int fsDrawID;

// Unjittered
out vec4 fsCurrentClipPos;
out vec4 fsPrevClipPos;

void main() {
    vec4 pos = modelMatrices[gl_DrawID] * vec4(getPosition(gl_VertexID), 1.0);
    vec4 clip = projectionView * pos;
    //clip.xy += jitter * clip.w;
    
    fsTexCoords = getTexCoord(gl_VertexID);
    fsDrawID = gl_DrawID;

    fsPrevClipPos = prevProjectionView * prevModelMatrices[gl_DrawID] * vec4(getPosition(gl_VertexID), 1.0);
    fsCurrentClipPos = clip;

    clip = jitterProjectionView * pos;
    gl_Position = clip;
}