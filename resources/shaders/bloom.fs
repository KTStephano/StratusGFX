#version 410 core

#define PREVENT_DIV_BY_ZERO 0.00001

uniform bool downsamplingStage = true;
uniform bool gaussianStage     = false;
uniform bool horizontal        = false;
uniform bool upsamplingStage   = false; // Performs upsample + previous bloom combine
uniform bool finalStage        = false;

uniform sampler2D mainTexture;
uniform sampler2D bloomTexture;

uniform float bloomThreshold = 0.45;
uniform float upsampleRadiusScale = 0.5;

uniform float viewportX;
uniform float viewportY;

#define WEIGHT_LENGTH 4

// Notice that the weights decrease, which signifies the start (weight[0]) contributing most
// and last (weight[4]) contributing least
//uniform float weights[WEIGHT_LENGTH] = float[] (0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
// See https://computergraphics.stackexchange.com/questions/39/how-is-gaussian-blur-implemented
uniform float weights[WEIGHT_LENGTH] = float[] (41.0 / 200.0, 26.0 / 200.0, 7.0 / 200.0, 3.0 / 200.0);

in vec2 fsTexCoords;
out vec3 fsColor;

// Prevents HDR color values from exceeding 16-bit color buffer range
vec3 boundHDR(vec3 value) {
    return min(value, 65500.0);
}

vec2 convertTexCoords(vec2 uv) {
    return uv; //(uv + 1.0) * 0.5;
}

vec2 computeTexelWidth(sampler2D tex, int miplevel) {
    // This will give us the size of a single texel in (x, y) directions
    // (miplevel is telling it to give us the size at mipmap *miplevel*, where 0 would mean full size image)
    return (1.0 / textureSize(tex, miplevel));// * vec2(2.0, 1.0);
}

vec2 screenTransformTexCoords(vec2 uv) {
    return uv;// * (viewportX / viewportY);
}

// Performs bilinear downsampling using 13 texel fetches
// See http://www.iryoku.com/next-generation-post-processing-in-call-of-duty-advanced-warfare for full presentation
// See https://www.youtube.com/watch?v=tI70-HIc5ro for explanation
// See https://catlikecoding.com/unity/tutorials/advanced-rendering/bloom/ for additional explanation
// See https://github.com/Unity-Technologies/Graphics/blob/master/com.unity.postprocessing/PostProcessing/Shaders/Builtins/Bloom.shader for reference implementation
vec3 downsampleBilinear13(sampler2D tex, vec2 uv) {
    vec2 texelWidth = computeTexelWidth(tex, 0);

    // Innermost square including center coordinate
    vec3 innermost       = texture(tex, screenTransformTexCoords(uv)).rgb;
    vec3 innerUpperLeft  = texture(tex, screenTransformTexCoords(uv + texelWidth * vec2(-0.5, 0.5))).rgb;
    vec3 innerLowerLeft  = texture(tex, screenTransformTexCoords(uv + texelWidth * vec2(-0.5, -0.5))).rgb;
    vec3 innerUpperRight = texture(tex, screenTransformTexCoords(uv + texelWidth * vec2(0.5, 0.5))).rgb;
    vec3 innerLowerRight = texture(tex, screenTransformTexCoords(uv + texelWidth * vec2(0.5, -0.5))).rgb;

    // Outermost square - notice the number of outer variables + number of inner variables = 13, which comprise
    // our 13 sampling points
    vec3 outerUpperLeft  = texture(tex, screenTransformTexCoords(uv + texelWidth * vec2(-1.0, 1.0))).rgb;
    vec3 outerLeft       = texture(tex, screenTransformTexCoords(uv + texelWidth * vec2(-1.0, 0.0))).rgb;
    vec3 outerLowerLeft  = texture(tex, screenTransformTexCoords(uv + texelWidth * vec2(-1.0, -1.0))).rgb;
    vec3 outerUpperRight = texture(tex, screenTransformTexCoords(uv + texelWidth * vec2(1.0, 1.0))).rgb;
    vec3 outerRight      = texture(tex, screenTransformTexCoords(uv + texelWidth * vec2(1.0, 0.0))).rgb;
    vec3 outerLowerRight = texture(tex, screenTransformTexCoords(uv + texelWidth * vec2(1.0, -1.0))).rgb;
    vec3 outerTop        = texture(tex, screenTransformTexCoords(uv + texelWidth * vec2(0.0, 1.0))).rgb;
    vec3 outerBottom     = texture(tex, screenTransformTexCoords(uv + texelWidth * vec2(0.0, -1.0))).rgb;

    // Comes from the weights in the CoD presentation
    vec2 weights = (1.0 / 4.0) * vec2(0.5, 0.25);

    return weights.x * (innerUpperLeft + innerLowerLeft + innerUpperRight + innerLowerRight) + // red circles in presentation
           weights.y * (outerUpperLeft + outerTop + outerLeft + innermost) +                   // yellowish circles in presentation
           weights.y * (outerLowerLeft + outerLeft + outerBottom + innermost) +                // purple circles in presentation
           weights.y * (outerBottom + outerLowerRight + outerRight + innermost) +              // blueish circles in presentation
           weights.y * (outerTop + outerUpperRight + outerRight + innermost);                  // green circles in presentation
}

// The idea here is we have a 3x3 filter kernel and we then perform 9 texel fetches (center, 3 upper, 3 lower, 2 side) and
// apply the kernel to the sampled texels
//
// The filter kernel looks like this:
//
//  (1/16) * [1 2 1]
//           [2 4 2]
//           [1 2 1]
//
// You can see the tent shape where the center is biggest, then the ones along + are next biggest, and ones on outer X are smallest
//
// Important: repeated convolutions converge to Gaussian, so we won't have any explicit Gaussian Blur code
vec3 upsampleTentFilter9(sampler2D tex, vec2 uv, float radiusScale) {
    vec2 texelWidth = computeTexelWidth(tex, 0);

    vec3 center     = texture(tex, screenTransformTexCoords(uv)).rgb;
    vec3 top        = texture(tex, screenTransformTexCoords(uv + texelWidth * vec2(0.0, 1.0)   * radiusScale)).rgb;
    vec3 upperLeft  = texture(tex, screenTransformTexCoords(uv + texelWidth * vec2(-1.0, 1.0)  * radiusScale)).rgb;
    vec3 left       = texture(tex, screenTransformTexCoords(uv + texelWidth * vec2(-1.0, 0.0)  * radiusScale)).rgb;
    vec3 lowerLeft  = texture(tex, screenTransformTexCoords(uv + texelWidth * vec2(-1.0, -1.0) * radiusScale)).rgb;
    vec3 bottom     = texture(tex, screenTransformTexCoords(uv + texelWidth * vec2(0.0, -1.0)  * radiusScale)).rgb;
    vec3 lowerRight = texture(tex, screenTransformTexCoords(uv + texelWidth * vec2(1.0, -1.0)  * radiusScale)).rgb;
    vec3 right      = texture(tex, screenTransformTexCoords(uv + texelWidth * vec2(1.0, 0.0)   * radiusScale)).rgb;
    vec3 upperRight = texture(tex, screenTransformTexCoords(uv + texelWidth * vec2(1.0, 1.0)   * radiusScale)).rgb;

    return (1.0 / 16.0) * (upperLeft + top * 2    + upperRight +
                       left * 2  + center * 4 + right * 2  +
                       lowerLeft + bottom * 2 + lowerRight);
}

float max3(vec3 vals) {
    return max(vals.r, max(vals.g, vals.b));
}

// Takes only the pixels which exceed the threshold
// See https://learnopengl.com/Advanced-Lighting/Bloom
vec3 filterBrightest(vec3 color, float threshold) {
    // TODO: Might need to find a better brightness filter
    // vec3 weights = vec3(0.2126, 0.7152, 0.0722);
    // float brightness = dot(color - vec3(threshold), weights); // multiplies each element together and sums them up
    // return max(vec3(0.0), color * brightness);
    // if (brightness > threshold) {
    //     return color;
    // }
    // else {
    //     return vec3(0.0);
    // }

    // color = clamp(color, 0.0, 30.0);

    float softening = 0.1;
    float offset = threshold * softening + PREVENT_DIV_BY_ZERO;    
    vec3 curve = vec3(threshold, threshold - offset, 0.25 / offset);

    // Pixel brightness
    float br = max3(color);

    // Under-threshold part: quadratic curve
    float rq = clamp(br - curve.x, 0.0, curve.y);
    rq = curve.z * rq * rq;

    // Combine and apply the brightness response curve.
    color *= max(rq, br - threshold) / max(br, PREVENT_DIV_BY_ZERO);

    return color;
}

// See https://learnopengl.com/Advanced-Lighting/Bloom
vec3 applyGaussianBlur() {
    vec3 color = texture(mainTexture, fsTexCoords).rgb * weights[0];
    // This will give us the size of a single texel in (x, y) directions
    // (the 0 is telling it to give us the size at mipmap 0, aka full size image)
    vec2 texelWidth = computeTexelWidth(mainTexture, 0);
    if (horizontal) {
        for (int i = 1; i < WEIGHT_LENGTH; ++i) {
            vec2 texOffset = vec2(texelWidth.x * i, 0.0);
            // Notice we to +- texOffset so we can calculate both directions at once from the starting pixel
            color = color + texture(mainTexture, fsTexCoords + texOffset).rgb * weights[i];
            color = color + texture(mainTexture, fsTexCoords - texOffset).rgb * weights[i];
        }
    }
    else {
        for (int i = 1; i < WEIGHT_LENGTH; ++i) {
            vec2 texOffset = vec2(0.0, texelWidth.y * i);
            color = color + texture(mainTexture, fsTexCoords + texOffset).rgb * weights[i];
            color = color + texture(mainTexture, fsTexCoords - texOffset).rgb * weights[i];
        }
    }

    return boundHDR(color);
}

// As per the CoD presentation, we apply filtering after downsampling
vec3 applyDownsampleStage() {
    vec3 color = downsampleBilinear13(mainTexture, convertTexCoords(fsTexCoords));
    //return boundHDR(color);
    return boundHDR(filterBrightest(color, bloomThreshold));
}

// As per the CoD presentation, we apply filtering after upsampling
// TODO: Unity doesn't seem to filter at this step - why??
//
// The upsample + bloom conceptually looks like this:
// Original image = O
// Downsample(O) = A
// Downsample(A) = B
// Downsample(B) = C
// Downsample(C) = D
// Upsample(D) + C = E
// Upsample(E) + B = F
// Upsample(F) + A = G
// At the end, resolution(A) == resolution(G)
//
// After this step the final thing to do is to add G to O to get the finished image with
// bloom fully applied.
vec3 applyUpsampleStage() {
    vec3 color = upsampleTentFilter9(mainTexture, convertTexCoords(fsTexCoords), upsampleRadiusScale);
    //return boundHDR(filterBrightest(color, bloomThreshold)); // --> no to this??
    vec3 bloom = texture(bloomTexture, fsTexCoords).rgb;
    color = color + bloom;
    return boundHDR(color);
    // if (!finalStage) {
    //     return boundHDR(filterBrightest(color, bloomThreshold));
    // }
    // else {
    //     return boundHDR(color);
    // }
}

void main() {
    if (downsamplingStage) {
        fsColor = applyDownsampleStage();
    }
    else if (gaussianStage) {
        fsColor = applyGaussianBlur();
    }
    else {
        fsColor = applyUpsampleStage();
    }
}