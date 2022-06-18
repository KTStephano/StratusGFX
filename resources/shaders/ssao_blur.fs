#version 410 core

smooth in vec2 fsTexCoords;

// ssao.vs should be used for its vertex shader

uniform sampler2DRect structureBuffer;
uniform sampler2DRect occlusionBuffer;

uniform float windowWidth;
uniform float windowHeight;

// GBuffer output
layout (location = 0) out float gLightFactor;

// This function averages 4 adjacent pixels. If bilinear filtering is turned on (GL_LINEAR) for the
// occlusion texture, this average becomes 16 total pixels with hardware averaging.
float blurAmbientOcclusion(vec2 pixelCoords) {
    // Small offset applied to max depth calculation
    const float deltaDepth = 1.0 / 128.0;
    const float deltaP = 1.5;

    vec4 structure = texture(structureBuffer, pixelCoords);
    // Calculate maximum depth offset. We will make comparisons between the depths of the different pixels,
    // and if max offset is exceeded we assume it is because a different piece of geometry has been reached.
    float range = deltaP * (max(abs(structure.x), abs(structure.y)) + deltaDepth);
    // Reconstruct depth of current center pixel
    float z0 = structure.z + structure.w;

    vec2 sampleData = vec2(0.0, 1.0);
    vec3 occlusion = vec3(0.0);
    for (int j = 0; j < 2; ++j) {
        // We want to sample at offsets -0.5 and 1.5
        float y = j * 2.0 - 0.5;
        
        for (int i = 0; i < 2; ++i) {
            // We want to sample at offsets -0.5 and 1.5
            float x = i * 2.0 - 0.5;

            // We are going to sample the change in X and accumulate the values
            vec2 sampleCoords = pixelCoords + vec2(x, y);
            sampleData.x = texture(occlusionBuffer, sampleCoords).x;
            // occlusion.z stores the total of all 4 occlusion values regardless of whether
            // they passed the depth test or not
            occlusion.z += sampleData.x;

            // Reconstruct the depth and if it is in range then accumulate it as well
            vec2 depth = texture(structureBuffer, sampleCoords).zw;
            float z = depth.x + depth.y;
            if (abs(z - z0) < range) {
                // occlusion.y holds # of samples that passed the depth test (sampleData.y is always 1.0)
                occlusion.xy += sampleData;
            }
        }
    }

    // If more than 1 sample passed the depth test, perform an average with just the ones that passed. If not,
    // we compensate by just averaging the value of all 4.
    return ((occlusion.y > 0.0) ? occlusion.x / occlusion.y : 0.25 * occlusion.z);
}

void main() {
    gLightFactor = blurAmbientOcclusion(fsTexCoords * vec2(windowWidth, windowHeight));
}