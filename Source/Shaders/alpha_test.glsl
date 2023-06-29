STRATUS_GLSL_VERSION

#define ALPHA_DEPTH_TEST_THRESHOLD 0.5
#define ALPHA_DEPTH_OFFSET 0.000001

// See "Implementing a material system" in 3D Graphics Rendering Cookbook
// This *only* uses basic punch through transparency and is not a full transparency solution
void runAlphaTest(float alpha) {
    if (ALPHA_DEPTH_TEST_THRESHOLD == 0.0) return;

    mat4 thresholdMatrix = mat4(
        1.0/17.0, 9.0/17.0, 3.0/17.0, 11.0/17.0,
        13.0/17.0, 5.0/17.0, 15.0/17.0, 7.0/17.0,
        4.0/17.0, 12.0/17.0, 2.0/17.0, 10.0/17.0,
        16.0/17.0, 8.0/17.0, 14.0/17.0, 6.0/17.0
    );

    int x = int(mod(gl_FragCoord.x, 4.0));
    int y = int(mod(gl_FragCoord.y, 4.0));
    alpha = clamp(alpha - 0.5 * thresholdMatrix[x][y], 0.0, 1.0);
    if (alpha < ALPHA_DEPTH_TEST_THRESHOLD) discard;
}