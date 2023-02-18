// This pass is responsible for per-pixel conversion to a luminance value
// using an equation found here: https://stackoverflow.com/questions/596216/formula-to-determine-perceived-brightness-of-rgb-color

// For implementation details see https://catlikecoding.com/unity/tutorials/advanced-rendering/fxaa/

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
uniform float subpixelBlending = 0.75;

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
    float blendFactor = smoothstep(0.0, 1.0, normalizedCenterAvgContrast);
    blendFactor = blendFactor * blendFactor * subpixelBlending;

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

    if (positiveGradient < negativeGradient) {
        pixelStep = -pixelStep;
    }

    vec2 uv = fsTexCoords;
    if (isHorizontal) {
        uv.y += pixelStep * blendFactor;
    }
    else {
        uv.x += pixelStep * blendFactor;
    }

    color = vec4(texture(screen, uv).rgb, 1.0);
}