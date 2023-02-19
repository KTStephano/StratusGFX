// This pass is responsible for per-pixel conversion to a luminance value
// using an equation found here: https://stackoverflow.com/questions/596216/formula-to-determine-perceived-brightness-of-rgb-color

// For implementation details see https://catlikecoding.com/unity/tutorials/advanced-rendering/fxaa/

// Summary of the algorithm:
//      BEGIN SECTION ONLY DEALING WITH IMMEDIATE 3x3 NEIGHBORHOOD OF PIXELS
//
//      1. Sample the luminance value at the current (center) pixel
//      2. Sample the luminance at its north, south, east and west neighbors
//      3. Calculate the minimum and maximum luminance values of the 5 pixels
//      4. Subtract max - min luminance to obtain an initial contrast value
//      5. If the initial contrast value is too small (min and max brightness is very similar)
//         then we skip the rest of the steps by assuming we are NOT on an edge.
//      6. Sample the remaining northeast, northwest, southeast and southwest neighbor pixel
//         luminance values.
//      7. Calculate the average luminance of the 8 neighbors (skipping center pixel) and multiply
//         luminance of north, south, east and west neighbors by 2 so that they are emphasized more heavily in the average.
//      8. Calculate a new contrast value by subtracting the average luminance by the middle luminance (convert negative values to positive).
//      9. Normalize the new contrast value by dividing it by the original max - min contrast value, and clamp the result between
//         0 and 1.
//      10. Calculate the first of two blend factors which is defined as:
//          smoothstep(0.0, 1.0, normalizedCenterAvgContrast) * smoothstep(0.0, 1.0, normalizedCenterAvgContrast) * subpixelBlending
//          where subpixelBlending is a configurable value between 0.0 (off) and 1.0 (max). We use smoothstep to perform a smooth
//          blending rather than a harsh blending if we just used the raw normalizedCenterAvgContrast value.
//      11. Now we need to determine if we are on a horizontal or vertical edge. To do this:
//          horizontal = abs sum of upper - center and lower - center contrast values (northwest - center, north - center, northeast - center,
//                                                                                     southwest - center, south - center, southeast - center)
//
//          vertical = abs sum of left - center and right - center contrast values (northwest - center, left - center, northeast - center,
//                                                                                  southwest - center, right - center, southeast - center)
//
//      12. If horizontal >= vertical, we're on a horizontal edge. Otherwise the edge is vertical.
//      13. Now compute the positive and negative contrast of the edge by abs value subtracting one side of the edge - center and the other side of the edge - center.
//      14. If positive contrast <= negative contrast, we need to move towards the negative side of the edge.
//      15. If desirable, end early and select the second pixel on either the positive or negative side of the edge and blend it with the center pixel.
//
//      BEGIN SECTION DEALING WITH LENGTH OF THE EDGE RATHER THAN JUST 3x3 LOCAL NEIGHBORHOOD
//
//      16. It is possible to use the information we have now to select one pixel from either the positive or negative side of the edge and blend it with
//          the center pixel to give us some level of anti-aliasting. However, this means only features defined within the 3x3 grid of neighborhood pixels
//          can be accounted for and anti-aliased. The rest of the algorithm is for walking along the edge in both directions to figure out how close the center pixel is to either
//          end of the edge and select a new blend weight based on this. A result of this is better feature detection so that we can anti-alias more fully along the
//          edge rather than just the 3x3 local neighborhood.
//      17. 

STRATUS_GLSL_VERSION

#include "common.glsl"

in vec2 fsTexCoords;

uniform sampler2D screen;

// Suggested values are as follows from the original FXAA algorithm:
//      0.0833 - upper limit
//      0.0625 - high quality
//      0.0312 - visible limit
uniform float contrastThreshold = 0.0625;

// Concept of relative threshold is that it is scaled by the maximum luminance
// of the local neighborhood of pixels. What it effectively does is ensure that
// when there is a region of high brightness, the contrast must be similarly high
// in order for anti-aliasing to be applied.
// Suggested values:
//      0.333 - to little (faster)
//      0.250 - low quality
//      0.166 - default
//      0.125 - high quality
//      0.063 - overkill (slower)
uniform float relativeThreshold = 0.125;

// The final strength of the blending can produce an image that is too blurry for what
// we might prefer. The original algorithm allowed us to specify a sub-pixel blending
// factor to reduce the strength of the blurring effect.
// Suggested values:
//      1.00 - upper limit (softer)
//      0.75 - default amount of filtering
//      0.50 - lower limit (sharper, less sub-pixel aliasing removal)
//      0.25 - almost off
//      0.00 - completely off
uniform float subpixelBlending = 1.0;

out vec4 color;

void main() {
    vec4 screenColor = texture(screen, fsTexCoords);

    // First sample the center luminance and its four neighbors in + directions
    float lumaCenter = screenColor.a;
    float lumaRight  = textureOffset(screen, fsTexCoords, ivec2( 1,  0)).a;
    float lumaLeft   = textureOffset(screen, fsTexCoords, ivec2(-1,  0)).a;
    float lumaTop    = textureOffset(screen, fsTexCoords, ivec2( 0,  1)).a;
    float lumaBot    = textureOffset(screen, fsTexCoords, ivec2( 0, -1)).a;

    // Now calculate the min and max luminance values
    float lumaMax = max(max(max(max(lumaCenter, lumaRight), lumaLeft), lumaTop), lumaBot);
    float lumaMin = min(min(min(min(lumaCenter, lumaRight), lumaLeft), lumaTop), lumaBot);

    // Now we subtract the two to get a contrast value
    //
    // High contrast means there is a large difference in perceptual brightness between the
    // brightest and dimmest pixels
    //
    // Low contrast means that the brightest and dimmest pixels are very similar perceptually
    float contrast = lumaMax - lumaMin;

    // Now check the contrast against the threshold where if it falls below (too little contrast)
    // we reject the sample and perform no anti-aliasing
    //
    // This process is effectively a basic edge detector
    float threshold = max(contrastThreshold, relativeThreshold * lumaMax);
    if (contrast < threshold) {
        color = vec4(screenColor.rgb, 1.0);
        return;
    }

    // If this sample passed the contrast threshold test, sample the rest of its neighbors
    float lumaTopRight = textureOffset(screen, fsTexCoords, ivec2( 1,  1)).a;
    float lumaBotRight = textureOffset(screen, fsTexCoords, ivec2( 1, -1)).a;
    float lumaTopLeft  = textureOffset(screen, fsTexCoords, ivec2(-1,  1)).a;
    float lumaBotLeft  = textureOffset(screen, fsTexCoords, ivec2(-1, -1)).a;

    // Perform a neighborhood average where neighbors along + are weighted more heavily (factor of 2)
    // than neighbors along the diagonal
    //
    // (We skip the center pixel for this)
    float average = 2.0 * (lumaRight + lumaLeft + lumaTop + lumaBot) +
                    lumaTopRight + lumaBotRight + lumaTopLeft + lumaBotLeft;
    // We divide by 12 instead of 8 since lumaRight, Left, Top and Bot each count for 2
    average = average / 12.0;

    // Now check the contrast between the center pixel and its neighborhood average perceptual luminance
    float centerAvgContrast = abs(average - lumaCenter);

    // Now normalize the contrast by dividing by the high - low contrast from the left, right, top, and bottom neighbors
    // (saturate clamps it between 0 and 1)
    float normalizedCenterAvgContrast = saturate(centerAvgContrast / contrast);

    // Use smoothstep on the normalized center average contrast so that the transition isn't as harsh as it
    // would be otherwise
    float pixelBlendFactor = smoothstep(0.0, 1.0, normalizedCenterAvgContrast);
    pixelBlendFactor = pixelBlendFactor * pixelBlendFactor * subpixelBlending;

    // For the next step we need to select the blend direction. FXAA takes the center pixel and blends it with
    // one of its 4 neighbors from the left, right, top and bottom.
    //
    // To do this we will use more contrast values. We are going off the insight that if there is a horizontal edge,
    // if we take the contrast between the upper and lower neighbors, there will be a large contrast. The same applies
    // to a vertical edge only this time we take the contrast between the left and right neighbors.
    //
    // We are including all 8 neighbors here in order to improve our edge direction detection.
    float horizontal = abs(lumaTop      + lumaBot      - 2.0 * lumaCenter) * 2.0 +
                       abs(lumaTopRight + lumaBotRight - 2.0 * lumaRight) +
                       abs(lumaTopLeft  + lumaBotLeft  - 2.0 * lumaLeft);

    float vertical   = abs(lumaRight    + lumaLeft     - 2.0 * lumaCenter) * 2.0 +
                       abs(lumaTopRight + lumaTopLeft  - 2.0 * lumaTop) + 
                       abs(lumaBotRight + lumaBotLeft  - 2.0 * lumaBot);

    // If the horizontal contrast is greater than vertical (meaning there is a sharper change in brightness along horizontal
    // vs vertical), then we say that we are on a horizontal edge.
    bool isHorizontal = horizontal >= vertical;

    // If it's a horizontal edge then we will perform a vertical blend
    // Otherwise we will perform a horizontal blend
    //
    // This is the width of an individual texel rather than the texture dimensions (second parameter is mip level where 0
    // is full size)
    float pixelStep = isHorizontal ? computeTexelSize(screen, 0).y : computeTexelSize(screen, 0).x;

    // Now we will select two neighboring luminance values based on whether it is horizontal or not.
    // If horizontal, positve luminance is equal to the top value. Otherwise it is equal to the right value.
    // If horizontal, negative luminance is equal to the bottom value. Otherwise it is equal to the left value.
    //
    // What we do then it subtract these two values from the center luminance to form a contrast gradient. If
    // the positive gradient is larger, our pixel step will be positive meaning we go in the positive direction for
    // blending. Otherwise we reverse the pixel step to go in the negative direction for blending.
    float positiveLuma = isHorizontal ? lumaTop : lumaRight;
    float negativeLuma = isHorizontal ? lumaBot : lumaLeft;
    float positiveGradient = abs(positiveLuma - lumaCenter);
    float negativeGradient = abs(negativeLuma - lumaCenter);
    float gradient;
    float oppositeLuma;

    if (positiveGradient < negativeGradient) {
        pixelStep = -pixelStep;
        oppositeLuma = negativeLuma;
        gradient = negativeGradient;
    }
    else {
        oppositeLuma = positiveLuma;
        gradient = positiveGradient;
    }

    vec2 uv = fsTexCoords;
    // This represents the coords which are halfway between two pixels - one on each side of the edge
    vec2 uvEdge = uv;
    // If uvEdge moves along +/- y direction, this will move along +/- x direction and vice versa. The reason is that
    // if uvEdge samples above and below us, we then want to move in the <-, -> directions and continuously sample
    // above and below. Ex: step -> to the right, sample above and below; step -> to the right, sample above and below, repeat until max steps.
    vec2 edgeStep;
    
    // The step along the edge uses 0.5 as a multiplier so that it is halfway between
    // two pixels (one on each side of the edge). With bilinear filtering this means we will get the average luminance between
    // them without having to explicitly sample both pixels in the pair.
    if (isHorizontal) {
        uvEdge.y += pixelStep * 0.5;
        edgeStep = vec2(computeTexelSize(screen, 0).x, 0.0);
    }
    else {
        uvEdge.x += pixelStep * 0.5;
        edgeStep = vec2(0.0, computeTexelSize(screen, 0).y);
    }

    float edgeLumaAvg = (lumaCenter + oppositeLuma) * 0.5;
    float gradientThreshold = gradient * 0.25;

    // We have arrays of 2 since we are going to step towards both the positive and negative ends
    // of the edge relative to the center pixel
    bool atEdgeEnds[2] = bool[](false, false);
    float edgeToCenterDistances[2] = float[](0.0, 0.0);
    float edgeLumaDeltas[2] = float[](0.0, 0.0);
    float directionalSigns[2] = float[](1.0, -1.0);
    vec2 edgeSteps[2] = vec2[](edgeStep, -edgeStep);

    for (int i = 0; i < 2; ++i) {
        vec2 puv = uvEdge + edgeSteps[i];
        float edgeLumaDelta = texture(screen, puv).a - edgeLumaAvg;
        // As soon as the contrast between current edge pixel and the edge average exceeds
        // the gradient threshold, we assume that to be the end of the edge
        bool atEdgeEnd = abs(edgeLumaDelta) >= gradientThreshold;

        // Perform 9 additional steps along the edge for a total of 10 each direction
        for (int j = 0; j < 9 && !atEdgeEnd; ++j) {
            puv += edgeSteps[i];
            edgeLumaDelta = texture(screen, puv).a - edgeLumaAvg;
            atEdgeEnd = abs(edgeLumaDelta) >= gradientThreshold;
        }

        edgeLumaDeltas[i] = edgeLumaDelta;
        atEdgeEnds[i] = atEdgeEnd;

        float distance;
        if (isHorizontal) {
            edgeToCenterDistances[i] = (directionalSigns[i] * puv.x) + (-directionalSigns[i] * uv.x);
        }
        else {
            edgeToCenterDistances[i] = (directionalSigns[i] * puv.y) + (-directionalSigns[i] * uv.y);
        }
    }

    // Determine the shortest distance (meaning along which direction of the edge our center pixel is closest to)
    // and then check to see that the difference between the end luma and average luma is positive or negative
    float shortestDistance;
    bool deltaSign;
    if (edgeToCenterDistances[0] <= edgeToCenterDistances[1]) {
        shortestDistance = edgeToCenterDistances[0];
        deltaSign = edgeLumaDeltas[0] >= 0;
    }
    else {
        shortestDistance = edgeToCenterDistances[1];
        deltaSign = edgeLumaDeltas[1] >= 0;
    }

    // If the delta sign and the center - average edge luminance are moving in different directions, it means our stepping
    // algorithmn was actually moving away from the edge rather than along it. If this is the case we just skip blending
    // altogether
    float edgeBlendFactor;
    if (deltaSign == (lumaCenter - edgeLumaAvg >= 0)) {
        edgeBlendFactor = 0.0;
    }
    else {
        // This ensures that the closer the center pixel gets to the edge, the more we blend
        edgeBlendFactor = 0.5 - shortestDistance / (edgeToCenterDistances[0] + edgeToCenterDistances[1]);
    }

    //edgeBlendFactor = edgeBlendFactor * edgeBlendFactor;

    // This is the final application of FXAA based on above calculations
    float finalBlendFactor = max(pixelBlendFactor, edgeBlendFactor);
    if (isHorizontal) {
        uv.y += pixelStep * finalBlendFactor;
    }
    else {
        uv.x += pixelStep * finalBlendFactor;
    }

    color = vec4(texture(screen, uv).rgb, 1.0);
}