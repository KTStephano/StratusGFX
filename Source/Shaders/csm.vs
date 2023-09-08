STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require
#extension GL_ARB_sparse_texture2 : require

// Cascaded Shadow Maps
// See Foundations of Game Engine Development Volume 2 (section on cascaded shadow maps)
// See https://learnopengl.com/Advanced-Lighting/Shadows/Shadow-Mapping
// See http://hacksoflife.blogspot.com/2009/01/polygon-offset-and-shadow-mapping.html
// See https://github.com/OGRECave/ogre-next/issues/100

// Enables gl_Layer and gl_ViewportIndex in the vertex shader (no geometry shader required)
#extension GL_ARB_shader_viewport_layer_array : require

#include "mesh_data.glsl"
#include "common.glsl"
#include "vsm_common.glsl"

layout (std430, binding = CURR_FRAME_MODEL_MATRICES_BINDING_POINT) readonly buffer SSBO3 {
    mat4 modelMatrices[];
};

uniform mat4 shadowMatrix;
//uniform mat4 globalVsmShadowMatrix;

uniform int vsmClipMapIndex;

uniform vec3 lightDir;
uniform int depthLayer;
//out float fsTanTheta;
flat out int fsDrawID;
flat out int fsClipMapIndex;
smooth out vec2 fsTexCoords;
smooth out vec2 vsmTexCoords;
//smooth out float vsmDepth;

void main () {
	// Select which layer of the depth texture we will write to
	// (DEPTH_LAYER is defined in C++ code)
	gl_Layer = DEPTH_LAYER;

	fsDrawID = gl_DrawID;
	fsTexCoords = getTexCoord(gl_VertexID);

	vec3 position = getPosition(gl_VertexID);
	mat4 worldMatrix = modelMatrices[gl_DrawID];

	// Since dot(l, n) = cos(theta) when both are normalized, below should compute tan theta
	//fsTanTheta = 3.0 * tan(acos(dot(normalize(lightDir), getNormal(gl_VertexID))));
	vec4 worldPos = worldMatrix * vec4(position, 1.0);
	vec4 clipPos = shadowMatrix * worldPos;

	//vec4 globalVsmClipPos = globalVsmShadowMatrix * worldPos;
	vec4 globalVsmClipPos = vec4(vsmCalculateOriginClipValueFromWorldPos(worldPos.xyz, vsmClipMapIndex), 1.0);
	// Perspective divide
	globalVsmClipPos.xyz /= globalVsmClipPos.w;
	// Transform from [-1, 1] to [0, 1]
	globalVsmClipPos.xyz = globalVsmClipPos.xyz * 0.5 + vec3(0.5);
	
	vsmTexCoords = globalVsmClipPos.xy;
	//vsmDepth = globalVsmClipPos.z;
	//vsmDepth = clipPos.z * 0.5 + 0.5;

	fsClipMapIndex = vsmClipMapIndex;

	gl_Position = clipPos;
}