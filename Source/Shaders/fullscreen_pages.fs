STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require
#extension GL_ARB_sparse_texture2 : require

in vec2 fsTexCoords;

uniform sampler2DArray depth;

uniform float znear;
uniform float zfar;

out vec4 color;

// See http://glampert.com/2014/01-26/visualizing-the-depth-buffer/
float linearizeDepth(in vec2 uv)
{
    uvec4 value;
    ivec3 pixelCoords = ivec3(uv * (textureSize(depth, 0).xy - vec2(1.0)), 0.0);
    int status = sparseTexelFetchARB(depth, pixelCoords, 0, value);
    float depth = uintBitsToFloat(value.r);
    if (sparseTexelsResidentARB(status) == false) {
        return 0.0;
    }
    //return depth == 1.0 ? 1.0 : 0.0;
    //return (2.0 * znear) / (zfar + znear - depth * (zfar - znear));
    return depth;
}

void main() {
    // vec4 value;
    // int status = sparseTextureARB(depth, vec3(fsTexCoords, 0.0), value);
    // //color = vec4(vec3(texture(depth, vec3(fsTexCoords, 0.0)).r), 1.0);
    // vec3 result = vec3(sparseTexelsResidentARB(status) == true ? 1.0 : 0.0, 0.0, 0.0);
    // color = vec4(result, 1.0);

    color = vec4(vec3(linearizeDepth(fsTexCoords)), 1.0);
}