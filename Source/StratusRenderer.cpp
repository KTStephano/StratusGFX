
#include <StratusRenderer.h>
#include <iostream>
#include <StratusLight.h>
#include "StratusPipeline.h"
#include "StratusRenderer.h"
#include <math.h>
#include <cmath>
#include "StratusUtils.h"
#include "StratusMath.h"
#include "StratusLog.h"
#include "StratusResourceManager.h"

namespace stratus {
static void printGLInfo(const GFXConfig & config) {
    auto & log = STRATUS_LOG << std::endl;
    log << "==================== OpenGL Information ====================" << std::endl;
    log << "\tRenderer: "                         << config.renderer << std::endl;
    log << "\tVersion: "                          << config.version << std::endl;
    log << "\tMax draw buffers: "                 << config.maxDrawBuffers << std::endl;
    log << "\tMax combined textures: "            << config.maxCombinedTextures << std::endl;
    log << "\tMax cube map texture size: "        << config.maxCubeMapTextureSize << std::endl;
    log << "\tMax fragment uniform vectors: "     << config.maxFragmentUniformVectors << std::endl;
    log << "\tMax fragment uniform components: "  << config.maxFragmentUniformComponents << std::endl;
    log << "\tMax varying floats: "               << config.maxVaryingFloats << std::endl;
    log << "\tMax render buffer size: "           << config.maxRenderbufferSize << std::endl;
    log << "\tMax texture image units: "          << config.maxTextureImageUnits << std::endl;
    log << "\tMax texture size: "                 << config.maxTextureSize << std::endl;
    log << "\tMax vertex attribs: "               << config.maxVertexAttribs << std::endl;
    log << "\tMax vertex uniform vectors: "       << config.maxVertexUniformVectors << std::endl;
    log << "\tMax vertex uniform components: "    << config.maxVertexUniformComponents << std::endl;
    log << "\tMax viewport dims: "                << "(" << config.maxViewportDims[0] << ", " << config.maxViewportDims[1] << ")" << std::endl;
}

Renderer::Renderer(SDL_Window * window) {
    _window = window;
    //const int32_t maxGLVersion = 3;
    //const int32_t minGLVersion = 2;

    // Set the profile to core as opposed to immediate mode
    SDL_GL_SetAttribute(SDL_GLattr::SDL_GL_CONTEXT_PROFILE_MASK,
            SDL_GL_CONTEXT_PROFILE_CORE);
    // Set max/min version to be 3.2
    //SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, maxGLVersion);
    //SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, minGLVersion);
    // Enable double buffering
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

    // Create the gl context
    _context = SDL_GL_CreateContext(window);
    if (_context == nullptr) {
        STRATUS_ERROR << "[error] Unable to create a valid OpenGL context" << std::endl;
        _isValid = false;
        return;
    }

    // Init gl core profile using gl3w
    if (gl3wInit()) {
        STRATUS_ERROR << "[error] Failed to initialize core OpenGL profile" << std::endl;
        _isValid = false;
        return;
    }

    //if (!gl3wIsSupported(maxGLVersion, minGLVersion)) {
    //    STRATUS_ERROR << "[error] OpenGL 3.2 not supported" << std::endl;
    //    _isValid = false;
    //    return;
    //}

    // Query OpenGL about various different hardware capabilities
    _config.renderer = (const char *)glGetString(GL_RENDERER);
    _config.version = (const char *)glGetString(GL_VERSION);
    glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, &_config.maxCombinedTextures);
    glGetIntegerv(GL_MAX_CUBE_MAP_TEXTURE_SIZE, &_config.maxCubeMapTextureSize);
    glGetIntegerv(GL_MAX_FRAGMENT_UNIFORM_VECTORS, &_config.maxFragmentUniformVectors);
    glGetIntegerv(GL_MAX_RENDERBUFFER_SIZE, &_config.maxRenderbufferSize);
    glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS, &_config.maxTextureImageUnits);
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &_config.maxTextureSize);
    glGetIntegerv(GL_MAX_VERTEX_ATTRIBS, &_config.maxVertexAttribs);
    glGetIntegerv(GL_MAX_VERTEX_UNIFORM_VECTORS, &_config.maxVertexUniformVectors);
    glGetIntegerv(GL_MAX_DRAW_BUFFERS, &_config.maxDrawBuffers);
    glGetIntegerv(GL_MAX_FRAGMENT_UNIFORM_COMPONENTS, &_config.maxFragmentUniformComponents);
    glGetIntegerv(GL_MAX_VERTEX_UNIFORM_COMPONENTS, &_config.maxVertexUniformComponents);
    glGetIntegerv(GL_MAX_VARYING_FLOATS, &_config.maxVaryingFloats);
    glGetIntegerv(GL_MAX_VIEWPORT_DIMS, _config.maxViewportDims);

    printGLInfo(_config);
    _isValid = true;

    // Initialize the pipelines
    _state.geometry = std::unique_ptr<Pipeline>(new Pipeline({
        Shader{"../resources/shaders/pbr_geometry_pass.vs", ShaderType::VERTEX}, 
        Shader{"../resources/shaders/pbr_geometry_pass.fs", ShaderType::FRAGMENT}}));
    _state.shaders.push_back(_state.geometry.get());

    _state.forward = std::unique_ptr<Pipeline>(new Pipeline({
        Shader{"../resources/shaders/flat_forward_pass.vs", ShaderType::VERTEX}, 
        Shader{"../resources/shaders/flat_forward_pass.fs", ShaderType::FRAGMENT}}));
    _state.shaders.push_back(_state.forward.get());

    using namespace std;

    // Set up the hdr/gamma postprocessing shader
    _state.hdrGamma = std::unique_ptr<Pipeline>(new Pipeline({
        Shader{"../resources/shaders/hdr.vs", ShaderType::VERTEX},
        Shader{"../resources/shaders/hdr.fs", ShaderType::FRAGMENT}}));
    _state.shaders.push_back(_state.hdrGamma.get());

    // Set up the shadow preprocessing shader
    _state.shadows = std::unique_ptr<Pipeline>(new Pipeline({
        Shader{"../resources/shaders/shadow.vs", ShaderType::VERTEX},
        Shader{"../resources/shaders/shadow.gs", ShaderType::GEOMETRY},
        Shader{"../resources/shaders/shadow.fs", ShaderType::FRAGMENT}}));
    _state.shaders.push_back(_state.shadows.get());

    _state.lighting = std::unique_ptr<Pipeline>(new Pipeline({
        Shader{"../resources/shaders/pbr.vs", ShaderType::VERTEX},
        Shader{"../resources/shaders/pbr.fs", ShaderType::FRAGMENT}}));
    _state.shaders.push_back(_state.lighting.get());

    _state.bloom = std::unique_ptr<Pipeline>(new Pipeline({
        Shader{"../resources/shaders/bloom.vs", ShaderType::VERTEX},
        Shader{"../resources/shaders/bloom.fs", ShaderType::FRAGMENT}}));
    _state.shaders.push_back(_state.bloom.get());

    _state.csmDepth = std::unique_ptr<Pipeline>(new Pipeline({
        Shader{"../resources/shaders/csm.vs", ShaderType::VERTEX},
        Shader{"../resources/shaders/csm.gs", ShaderType::GEOMETRY},
        Shader{"../resources/shaders/csm.fs", ShaderType::FRAGMENT}}));
    _state.shaders.push_back(_state.csmDepth.get());

    // Create the screen quad
    _state.screenQuad = ResourceManager::Instance()->CreateQuad()->GetRenderNode();

    // Use the shader isValid() method to determine if everything succeeded
    _validateAllShaders();

    _state.dummyCubeMap = createShadowMap3D(_state.shadowCubeMapX, _state.shadowCubeMapY);

    // Create a pool of shadow maps for point lights to use
    for (int i = 0; i < _state.numShadowMaps; ++i) {
        createShadowMap3D(_state.shadowCubeMapX, _state.shadowCubeMapY);
    }
}

void Renderer::_validateAllShaders() {
    _isValid = true;
    for (Pipeline * p : _state.shaders) {
        _isValid = _isValid && p->isValid();
    }
}

Renderer::~Renderer() {
    if (_context) {
        SDL_GL_DeleteContext(_context);
        _context = nullptr;
    }
    for (Pipeline * shader : _shaders) delete shader;
    _shaders.clear();

    // Delete the main frame buffer
    _clearGBuffer();
}

void Renderer::recompileShaders() {
    for (Pipeline* p : _state.shaders) {
        p->recompile();
    }
    _validateAllShaders();
}

const GFXConfig & Renderer::config() const {
    return _config;
}

bool Renderer::valid() const {
    return _isValid;
}

void Renderer::setClearColor(const Color &c) {
    _state.clearColor = c;
}

const Pipeline *Renderer::getCurrentShader() const {
    return nullptr;
}

void Renderer::_recalculateProjMatrices() {
    _state.perspective = glm::perspective(Radians(_state.fov).value(),
            float(_state.windowWidth) / float(_state.windowHeight),
            _state.znear,
            _state.zfar);
    // arguments: left, right, bottom, top, near, far - this matrix
    // transforms [0,width] to [-1, 1] and [0, height] to [-1, 1]
    _state.orthographic = glm::ortho(0.0f, float(_state.windowWidth),
            float(_state.windowHeight), 0.0f, -1.0f, 1.0f);

    // Recalculate cascade-specific data
    _state.worldLightIsDirty = true;
}

void Renderer::_recalculateCascadeData(const Camera & c) {
    static constexpr int numCascades = 4;
    _state.worldLightIsDirty = false;
    static constexpr int cascadeResolutionXY = 4096;
    static constexpr float cascadeResReciprocal = 1.0f / cascadeResolutionXY;
    static constexpr float cascadeDelta = cascadeResReciprocal;

    // See "Foundations of Game Engine Development, Volume 2: Rendering (pp. 178)
    //
    // FOV_x = 2tan^-1(s/g), FOV_y = 2tan^-1(1/g)
    // ==> tan(FOV_y/2)=1/g ==> g=1/tan(FOV_y/2)
    // where s is the aspect ratio (width / height)
    //_state.csms.clear();
    const bool recalculateFbos = _state.csms.size() == 0;
    _state.csms.resize(numCascades);

    // Set up the shadow texture offsets
    // _state.cascadeShadowOffsets[0] = glm::vec4(-cascadeDelta, -2.0f * cascadeDelta, 2.0f * cascadeDelta, -cascadeDelta);
    // _state.cascadeShadowOffsets[1] = glm::vec4(cascadeDelta, 2.0f * cascadeDelta, -2.0f * cascadeDelta, cascadeDelta);
    _state.cascadeShadowOffsets[0] = glm::vec4(-cascadeDelta, -cascadeDelta, cascadeDelta, -cascadeDelta);
    _state.cascadeShadowOffsets[1] = glm::vec4(cascadeDelta, cascadeDelta, -cascadeDelta, cascadeDelta);

    // Assume directional light translation is none
    // Camera light(false);
    // light.setAngle(_state.worldLight.getRotation());
    const Camera & light = _state.worldLightCamera;
    const glm::mat4& lightWorldTransform = light.getWorldTransform();
    const glm::mat4& lightViewTransform = light.getViewTransform();
    const glm::mat4& cameraWorldTransform = c.getWorldTransform();
    const glm::mat4 transposeLightWorldTransform = glm::transpose(lightWorldTransform);

    const glm::mat4 L = lightViewTransform * cameraWorldTransform;

    const float s = float(_state.windowWidth) / float(_state.windowHeight);
    // g is also known as the camera's focal length
    const float g = 1.0 / tangent(_state.fov / 2.0f).value();
    const float znear = _state.znear;
    // We don't want zfar to be unbounded, so we constrain it to at most 600 which also has the nice bonus
    // of increasing our shadow map resolution (same shadow texture resolution over a smaller total area)
    const float zfar  = std::min(600.0f, _state.zfar);

    // @see https://johanmedestrom.wordpress.com/2016/03/18/opengl-cascaded-shadow-maps/
    // @see https://johanmedestrom.wordpress.com/2016/03/18/opengl-cascaded-shadow-maps/
    const float lambda = 0.5f;
    const float clipRange = zfar - znear;
    const float ratio = zfar / znear;
    std::vector<float> cascadeEnds(numCascades);
    for (int i = 0; i < numCascades; ++i) {
        // We are going to select the cascade split points by computing the logarithmic split, then the uniform split,
        // and then combining them by lambda * log + (1 - lambda) * uniform - the benefit is that it will produce relatively
        // consistent sampling depths over the whole frustum. This is in contrast to under or oversampling inconsistently at different
        // distances.
        const float p = (i + 1) / float(numCascades);
        const float log = znear * std::pow(ratio, p);
        const float uniform = znear + clipRange * p;
        const float d = std::floorf(lambda * (log - uniform) + uniform);
        cascadeEnds[i] = d;
    }

    // We offset each cascade begin from 1 onwards so that there is some overlap between the start of cascade k and the end of cascade k-1
    const std::vector<float> cascadeBegins = { 0.0f, cascadeEnds[0] - 10.0f,  cascadeEnds[1] - 10.0f, cascadeEnds[2] - 10.0f }; // 4 cascades max
    //const std::vector<float> cascadeEnds   = {  30.0f, 100.0f, 240.0f, 640.0f };
    std::vector<float> aks;
    std::vector<float> bks;
    std::vector<float> dks;
    std::vector<glm::vec3> sks;
    std::vector<float> zmins;
    std::vector<float> zmaxs;

    if (recalculateFbos) {
        // Create the depth buffer
        // @see https://stackoverflow.com/questions/22419682/glsl-sampler2dshadow-and-shadow2d-clarificationssss
        Texture tex(TextureConfig{ TextureType::TEXTURE_2D_ARRAY, TextureComponentFormat::DEPTH, TextureComponentSize::BITS_DEFAULT, TextureComponentType::FLOAT, cascadeResolutionXY, cascadeResolutionXY, numCascades, false }, nullptr);
        tex.setMinMagFilter(TextureMinificationFilter::NEAREST, TextureMagnificationFilter::NEAREST);
        tex.setCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);
        // We need to set this when using sampler2DShadow in the GLSL shader
        tex.setTextureCompare(TextureCompareMode::COMPARE_REF_TO_TEXTURE, TextureCompareFunc::LEQUAL);

        // Create the frame buffer
        _state.cascadeFbo = FrameBuffer({ tex });
    }

    for (int i = 0; i < numCascades; ++i) {
        const float ak = cascadeBegins[i];
        const float bk = cascadeEnds[i];
        aks.push_back(ak);
        bks.push_back(bk);

        // These base values are in camera space and define our frustum corners
        const float baseAkX = (ak * s) / g;
        const float baseAkY = ak / g;
        const float baseBkX = (bk * s) / g;
        const float baseBkY = bk / g;
        // Keep all of these in camera space for now
        std::vector<glm::vec4> frustumCorners = {
            // Near corners
            glm::vec4(baseAkX, -baseAkY, -ak, 1.0f),
            glm::vec4(baseAkX, baseAkY, -ak, 1.0f),
            glm::vec4(-baseAkX, baseAkY, -ak, 1.0f),
            glm::vec4(-baseAkX, -baseAkY, -ak, 1.0f),

            // Far corners
            glm::vec4(baseBkX, -baseBkY, -bk, 1.0f),
            glm::vec4(baseBkX, baseBkY, -bk, 1.0f),
            glm::vec4(-baseBkX, baseBkY, -bk, 1.0f),
            glm::vec4(-baseBkX, -baseBkY, -bk, 1.0f),
        };
        
        // This tells us the maximum diameter for the cascade bounding box
        const float dk = std::ceilf(std::max<float>(glm::length(frustumCorners[0] - frustumCorners[6]), 
                                                    glm::length(frustumCorners[4] - frustumCorners[6])));
        dks.push_back(dk);
        // T is essentially the physical width/height of area corresponding to each texel in the shadow map
        const float T = dk / cascadeResolutionXY;

        // Compute min/max of each so that we can combine it with dk to create a perfectly rectangular bounding box
        float minX = std::numeric_limits<float>::max();
        float maxX = std::numeric_limits<float>::min();
        float minY = minX;
        float maxY = maxX;
        float minZ = minX;
        float maxZ = maxX;
        for (int j = 0; j < frustumCorners.size(); ++j) {
            // First make sure to transform frustumCorners[j] from camera space to light space
            frustumCorners[j] = L * frustumCorners[j];

            minX = std::min(minX, frustumCorners[j].x);
            maxX = std::max(maxX, frustumCorners[j].x);

            minY = std::min(minY, frustumCorners[j].y);
            maxY = std::max(maxY, frustumCorners[j].y);

            minZ = std::min(minZ, frustumCorners[j].z);
            maxZ = std::max(maxZ, frustumCorners[j].z);
        }

        zmins.push_back(minZ);
        zmaxs.push_back(maxZ);

        // Now we calculate cascade camera position sk using the min, max, dk and T for a stable location
        glm::vec3 sk(std::floorf((maxX + minX) / (2.0f * T)) * T, 
                     std::floorf((maxY + minY) / (2.0f * T)) * T, 
                     minZ);
        //sk = glm::vec3(L * glm::vec4(sk, 1.0f));
        // STRATUS_LOG << "sk " << sk << std::endl;
        sks.push_back(sk);

        // We use transposeLightWorldTransform because it's less precision-error-prone than just doing glm::inverse(lightWorldTransform)
        // Note: we use -sk instead of lightWorldTransform * sk because we're assuming the translation component is 0
        const glm::mat4 cascadeViewTransform = glm::mat4(transposeLightWorldTransform[0], 
                                                         transposeLightWorldTransform[1],
                                                         transposeLightWorldTransform[2],
                                                         glm::vec4(-sk, 1.0f));

        // We add this into the cascadeOrthoProjection map to add a slight depth offset to each value which helps reduce flickering artifacts
        const float shadowDepthOffset = 2e-19;
        // We are putting the light camera location sk on the near plane in the halfway point between left, right, top and bottom planes
        // so it enables us to use the simplified Orthographic Projection matrix below
        //
        // This results in values between [-1, 1]
        const glm::mat4 cascadeOrthoProjection(glm::vec4(2.0f / dk, 0.0f, 0.0f, 0.0f), 
                                               glm::vec4(0.0f, 2.0f / dk, 0.0f, 0.0f),
                                               glm::vec4(0.0f, 0.0f, 1.0f / (maxZ - minZ), shadowDepthOffset),
                                               glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));

        // Gives us x, y values between [0, 1]
        const glm::mat4 cascadeTexelOrthoProjection(glm::vec4(1.0f / dk, 0.0f, 0.0f, 0.0f), 
                                                    glm::vec4(0.0f, 1.0f / dk, 0.0f, 0.0f),
                                                    glm::vec4(0.0f, 0.0f, 1.0f / (maxZ - minZ), 0.0f),
                                                    glm::vec4(0.5f, 0.5f, 0.0f, 1.0f));

        // Note: if we want we can set texelProjection to be cascadeTexelOrthoProjection and then set projectionView
        // to be cascadeTexelOrthoProjection * cascadeViewTransform. This has the added benefit of automatically translating
        // x, y positions to texel coordinates on the range [0, 1] rather than [-1, 1].
        //
        // However, the alternative is to just compute (coordinate * 0.5 + 0.5) in the fragment shader which does the same thing.
        _state.csms[i].depthProjection = cascadeOrthoProjection;
        _state.csms[i].texelProjection = cascadeOrthoProjection;
        _state.csms[i].view = cascadeViewTransform;
        _state.csms[i].projectionView = cascadeOrthoProjection * cascadeViewTransform;

        if (i > 0) {
            // This will allow us to calculate the cascade blending weights in the vertex shader and then
            // the cascade indices in the pixel shader
            const glm::vec3 n = -glm::vec3(cameraWorldTransform[2]);
            const glm::vec3 c = glm::vec3(cameraWorldTransform[3]);
            // fk now represents a plane along the direction of the view frustum. Its normal is equal to the camera's forward
            // direction in world space and it contains the point c + ak*n.
            const glm::vec4 fk = glm::vec4(n.x, n.y, n.z, glm::dot(-n, c) - ak) * (1.0f / (bks[i - 1] - ak));
            _state.csms[i].cascadePlane = fk;
        }
    }
}

void Renderer::_clearGBuffer() {
    _state.buffer = GBuffer();

    // for (auto postFx : _state.postFxBuffers) {
    //     glDeleteFramebuffers(1, &postFx.fbo);
    //     glDeleteTextures(1, &postFx.colorBuffer);
    //     glDeleteTextures(postFx.additionalBuffers.size(), &postFx.additionalBuffers[0]);
    // }
    _state.gaussianBuffers.clear();
    _state.postFxBuffers.clear();
}

void Renderer::_setWindowDimensions(int w, int h) {
    if (_state.windowWidth == w && _state.windowHeight == h) return;
    if (w < 0 || h < 0) return;
    _state.windowWidth = w;
    _state.windowHeight = h;
    _recalculateProjMatrices();
    glViewport(0, 0, w, h);

    // Regenerate the main frame buffer
    _clearGBuffer();

    GBuffer & buffer = _state.buffer;
    // glGenFramebuffers(1, &buffer.fbo);
    // glBindFramebuffer(GL_FRAMEBUFFER, buffer.fbo);

    // Position buffer
    buffer.position = Texture(TextureConfig{TextureType::TEXTURE_2D, TextureComponentFormat::RGB, TextureComponentSize::BITS_32, TextureComponentType::FLOAT, _state.windowWidth, _state.windowHeight, 0, false}, nullptr);
    buffer.position.setMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);

    // Normal buffer
    buffer.normals = Texture(TextureConfig{TextureType::TEXTURE_2D, TextureComponentFormat::RGB, TextureComponentSize::BITS_32, TextureComponentType::FLOAT, _state.windowWidth, _state.windowHeight, 0, false}, nullptr);
    buffer.normals.setMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);

    // Create the color buffer - notice that is uses higher
    // than normal precision. This allows us to write color values
    // greater than 1.0 to support things like HDR.
    buffer.albedo = Texture(TextureConfig{TextureType::TEXTURE_2D, TextureComponentFormat::RGB, TextureComponentSize::BITS_16, TextureComponentType::FLOAT, _state.windowWidth, _state.windowHeight, 0, false}, nullptr);
    buffer.albedo.setMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);

    // Base reflectivity buffer
    buffer.baseReflectivity = Texture(TextureConfig{TextureType::TEXTURE_2D, TextureComponentFormat::RGB, TextureComponentSize::BITS_16, TextureComponentType::FLOAT, _state.windowWidth, _state.windowHeight, 0, false}, nullptr);
    buffer.baseReflectivity.setMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);

    // Roughness-Metallic-Ambient buffer
    buffer.roughnessMetallicAmbient = Texture(TextureConfig{TextureType::TEXTURE_2D, TextureComponentFormat::RGB, TextureComponentSize::BITS_16, TextureComponentType::FLOAT, _state.windowWidth, _state.windowHeight, 0, false}, nullptr);
    buffer.roughnessMetallicAmbient.setMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);

    // Create the depth buffer
    buffer.depth = Texture(TextureConfig{TextureType::TEXTURE_2D, TextureComponentFormat::DEPTH, TextureComponentSize::BITS_DEFAULT, TextureComponentType::FLOAT, _state.windowWidth, _state.windowHeight, 0, false}, nullptr);
    buffer.depth.setMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);

    // Create the frame buffer with all its texture attachments
    buffer.fbo = FrameBuffer({buffer.position, buffer.normals, buffer.albedo, buffer.baseReflectivity, buffer.roughnessMetallicAmbient, buffer.depth});
    if (!buffer.fbo.valid()) {
        _isValid = false;
        return;
    }

    // Code to create the lighting fbo
    _state.lightingColorBuffer = Texture(TextureConfig{TextureType::TEXTURE_2D, TextureComponentFormat::RGB, TextureComponentSize::BITS_16, TextureComponentType::FLOAT, _state.windowWidth, _state.windowHeight, 0, false}, nullptr);
    _state.lightingColorBuffer.setMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);
    _state.lightingColorBuffer.setCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);

    // Create the buffer we will use to add bloom as a post-processing effect
    _state.lightingHighBrightnessBuffer = Texture(TextureConfig{TextureType::TEXTURE_2D, TextureComponentFormat::RGB, TextureComponentSize::BITS_16, TextureComponentType::FLOAT, _state.windowWidth, _state.windowHeight, 0, false}, nullptr);
    _state.lightingHighBrightnessBuffer.setMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);
    _state.lightingHighBrightnessBuffer.setCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);

    // Create the depth buffer
    _state.lightingDepthBuffer = Texture(TextureConfig{TextureType::TEXTURE_2D, TextureComponentFormat::DEPTH, TextureComponentSize::BITS_DEFAULT, TextureComponentType::FLOAT, _state.windowWidth, _state.windowHeight, 0, false}, nullptr);
    _state.lightingDepthBuffer.setMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);

    // Attach the textures to the FBO
    _state.lightingFbo = FrameBuffer({_state.lightingColorBuffer, _state.lightingHighBrightnessBuffer, _state.lightingDepthBuffer});
    if (!_state.lightingFbo.valid()) {
        _isValid = false;
        return;
    }

    _initializePostFxBuffers();
}

void Renderer::_initializePostFxBuffers() {
    uint32_t currWidth = _state.windowWidth;
    uint32_t currHeight = _state.windowHeight;
    _state.numDownsampleIterations = 0;
    _state.numUpsampleIterations = 0;

    // Initialize bloom
    for (; _state.numDownsampleIterations < 8; ++_state.numDownsampleIterations) {
        currWidth /= 2;
        currHeight /= 2;
        if (currWidth < 8 || currHeight < 8) break;
        PostFXBuffer buffer;
        auto color = Texture(TextureConfig{ TextureType::TEXTURE_2D, TextureComponentFormat::RGB, TextureComponentSize::BITS_16, TextureComponentType::FLOAT, currWidth, currHeight, 0, false }, nullptr);
        color.setMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);
        color.setCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE); // TODO: Does this make sense for bloom textures?
        buffer.fbo = FrameBuffer({ color });
        if (!buffer.fbo.valid()) {
            _isValid = false;
            STRATUS_ERROR << "Unable to initialize bloom buffer" << std::endl;
            return;
        }
        _state.postFxBuffers.push_back(buffer);

        // Create the Gaussian Blur buffers
        PostFXBuffer dualBlurFbos[2];
        for (int i = 0; i < 2; ++i) {
            FrameBuffer& blurFbo = dualBlurFbos[i].fbo;
            Texture tex = Texture(color.getConfig(), nullptr);
            tex.setMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);
            tex.setCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);
            blurFbo = FrameBuffer({tex});
            _state.gaussianBuffers.push_back(dualBlurFbos[i]);
        }
    }

    std::vector<std::pair<uint32_t, uint32_t>> sizes;
    for (int i = _state.numDownsampleIterations - 2; i >= 0; --i) {
        auto tex = _state.postFxBuffers[i].fbo.getColorAttachments()[0];
        sizes.push_back(std::make_pair(tex.width(), tex.height()));
    }
    sizes.push_back(std::make_pair(_state.windowWidth, _state.windowHeight));
    
    for (auto&[width, height] : sizes) {
        PostFXBuffer buffer;
        ++_state.numUpsampleIterations;
        auto color = Texture(TextureConfig{ TextureType::TEXTURE_2D, TextureComponentFormat::RGB, TextureComponentSize::BITS_16, TextureComponentType::FLOAT, width, height, 0, false }, nullptr);
        color.setMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);
        color.setCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE); // TODO: Does this make sense for bloom textures?
        buffer.fbo = FrameBuffer({ color });
        if (!buffer.fbo.valid()) {
            _isValid = false;
            STRATUS_ERROR << "Unable to initialize bloom buffer" << std::endl;
            return;
        }
        _state.postFxBuffers.push_back(buffer);
    }
}

void Renderer::setPerspectiveData(const Degrees & fov, float fnear, float ffar) {
    // TODO: Find the best lower bound for fov instead of arbitrary 25.0f
    if (fov.value() < 25.0f) return;
    _state.fov = fov;
    _state.znear = fnear;
    _state.zfar = ffar;
    _recalculateProjMatrices();
}

void Renderer::begin(bool clearScreen) {
    // Make sure we set our context as the active one
    SDL_GL_MakeCurrent(_window, _context);

    // Check for changes in the window size
    int w, h;
    SDL_GetWindowSize(_window, &w, &h);
    // This won't resize anything if the width/height
    // didn't change
    _setWindowDimensions(w, h);

    // Always clear the main screen buffer, but only
    // conditionally clean the custom frame buffer
    glClearColor(_state.clearColor.r, _state.clearColor.g,
                 _state.clearColor.b, _state.clearColor.a);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (clearScreen) {
        glm::vec4 color = glm::vec4(_state.clearColor.r, _state.clearColor.g, _state.clearColor.b, _state.clearColor.a);
        _state.buffer.fbo.clear(color);
        _state.lightingFbo.clear(color);

        // Depending on when this happens we may not have generated cascadeFbo yet
        if (_state.cascadeFbo.valid()) _state.cascadeFbo.clear(color);

        for (auto& gaussian : _state.gaussianBuffers) {
            gaussian.fbo.clear(color);
        }

        for (auto& postFx : _state.postFxBuffers) {
            postFx.fbo.clear(color);
        }
    }

    // Clear all entities from the previous frame
    _state.entities.clear();

    // Clear all instanced entities
    _state.instancedLightInteractMeshes.clear();
    _state.instancedFlatMeshes.clear();

    // Clear all lights
    _state.lights.clear();
    _state.lightInteractingEntities.clear();

    glDisable(GL_BLEND);
    //glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_CULL_FACE);
    glFrontFace(GL_CW);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LINE_SMOOTH);
    glEnable(GL_POLYGON_SMOOTH);
    // See https://paroj.github.io/gltut/Positioning/Tut05%20Depth%20Clamping.html
    glEnable(GL_DEPTH_CLAMP);

    // This is important! It prevents z-fighting if you do multiple passes.
    glDepthFunc(GL_LEQUAL);
}

void Renderer::addDrawable(const EntityPtr& e) {
    _addDrawable(e);
}

static void addEntityMeshData(const RenderNodeView& e, std::unordered_map<RenderNodeView, std::vector<__MeshInstanceContainer>> & map) {
    if (map.find(e) == map.end()) {
        std::vector<__MeshInstanceContainer> c;
        for (int i = 0; i < e.Get()->GetNumMeshContainers(); ++i) {
            c.push_back(__MeshInstanceContainer());
        }
        map.insert(std::make_pair(e, c));
    }

    std::vector<__MeshInstanceContainer>& existing = map.find(e)->second;
    for (int i = 0; i < e.Get()->GetNumMeshContainers(); ++i) {
        const RenderMeshContainer* c = e.Get()->GetMeshContainer(i);
        __MeshInstanceContainer& container = existing[i];
        container.modelMatrices.push_back(e.Get()->GetWorldTransform());
        container.diffuseColors.push_back(c->material->GetDiffuseColor());
        container.baseReflectivity.push_back(c->material->GetBaseReflectivity());
        container.roughness.push_back(c->material->GetRoughness());
        container.metallic.push_back(c->material->GetMetallic());
        ++container.size;
    }
}

void Renderer::_addDrawable(const EntityPtr& e) {
    if (e == nullptr) return;
    if (e->GetRenderNode() == nullptr || e->GetRenderNode()->GetInvisible()) {
        for (auto& child : e->GetChildren()) {
            _addDrawable(child);
        }
        return;
    }

    RenderNodeView rnode = RenderNodeView(e->GetRenderNode());
    _state.entities.push_back(rnode);
    if (rnode.Get()->GetLightInteractionEnabled()) {
        _state.lightInteractingEntities.push_back(rnode);
    }

    // We want to keep track of entities and whether or not they have moved for determining
    // when shadows should be recomputed
    if (_entitiesSeenBefore.find(EntityView(e)) == _entitiesSeenBefore.end()) {
        _entitiesSeenBefore.insert(std::make_pair(EntityView(e), EntityStateInfo{e->GetWorldPosition(), e->GetLocalScale(), e->GetLocalRotation().asVec3(), true}));
    }
    else {
        EntityStateInfo & info = _entitiesSeenBefore.find(EntityView(e))->second;
        const double distance = glm::distance(e->GetWorldPosition(), info.lastPos);
        const double scale = glm::distance(e->GetLocalScale(), info.lastScale);
        const double rotation = glm::distance(e->GetLocalRotation().asVec3(), info.lastRotation);
        info.dirty = false;
        if (distance > 0.25) {
            info.lastPos = e->GetWorldPosition();
            info.dirty = true;
        }
        if (scale > 0.25) {
            info.lastScale = e->GetLocalScale();
            info.dirty = true;
        }
        if (rotation > 0.25) {
            info.lastRotation = e->GetLocalRotation().asVec3();
            info.dirty = true;
        }
    }

    //addEntityMeshData(e, _state.instancedMeshes);

    for (auto& child : e->GetChildren()) {
        _addDrawable(child);
    }
}

/**
 * During the lighting phase, we need each of the 6 faces of the shadow map to have its own view transform matrix.
 * This enables us to convert vertices to be in various different light coordinate spaces.
 */
static std::vector<glm::mat4> generateLightViewTransforms(const glm::mat4 & projection, const glm::vec3 & lightPos) {
    return std::vector<glm::mat4>{
        //                       pos       pos + dir                                  up
        projection * glm::lookAt(lightPos, lightPos + glm::vec3( 1.0f,  0.0f,  0.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
        projection * glm::lookAt(lightPos, lightPos + glm::vec3(-1.0f,  0.0f,  0.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
        projection * glm::lookAt(lightPos, lightPos + glm::vec3( 0.0f,  1.0f,  0.0f), glm::vec3(0.0f,  0.0f,  1.0f)),
        projection * glm::lookAt(lightPos, lightPos + glm::vec3( 0.0f, -1.0f,  0.0f), glm::vec3(0.0f,  0.0f, -1.0f)),
        projection * glm::lookAt(lightPos, lightPos + glm::vec3( 0.0f,  0.0f,  1.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
        projection * glm::lookAt(lightPos, lightPos + glm::vec3( 0.0f,  0.0f, -1.0f), glm::vec3(0.0f, -1.0f,  0.0f))
    };
}

void Renderer::_initInstancedData(__MeshInstanceContainer & c, std::vector<GpuArrayBuffer> & gabuffers) {
    Pipeline * pbr = _state.geometry.get();

    auto & modelMats = c.modelMatrices;
    auto & diffuseColors = c.diffuseColors;
    auto & baseReflectivity = c.baseReflectivity;
    auto & roughness = c.roughness;
    auto & metallic = c.metallic;
    auto & buffers = c.buffers;
    buffers.Clear();
    gabuffers.push_back(buffers);

    GpuBuffer buffer;

    // First the model matrices

    // All shaders should use the same location for model, so this should work
    int pos = pbr->getAttribLocation("model");
    buffer = GpuBuffer(GpuBufferType::PRIMITIVE_BUFFER, modelMats.data(), modelMats.size() * sizeof(glm::mat4));
    buffer.EnableAttribute(pos, 16, GpuStorageType::FLOAT, false, 0, 0, 1);
    buffers.AddBuffer(buffer);

    pos = pbr->getAttribLocation("diffuseColor");
    buffer = GpuBuffer(GpuBufferType::PRIMITIVE_BUFFER, diffuseColors.data(), diffuseColors.size() * sizeof(glm::vec3));
    buffer.EnableAttribute(pos, 3, GpuStorageType::FLOAT, false, 0, 0);
    buffers.AddBuffer(buffer);

    pos = pbr->getAttribLocation("baseReflectivity");
    buffer = GpuBuffer(GpuBufferType::PRIMITIVE_BUFFER, baseReflectivity.data(), baseReflectivity.size() * sizeof(glm::vec3));
    buffer.EnableAttribute(pos, 3, GpuStorageType::FLOAT, false, 0, 0);
    buffers.AddBuffer(buffer);

    pos = pbr->getAttribLocation("metallic");
    buffer = GpuBuffer(GpuBufferType::PRIMITIVE_BUFFER, metallic.data(), metallic.size() * sizeof(float));
    buffer.EnableAttribute(pos, 1, GpuStorageType::FLOAT, false, 0, 0);
    buffers.AddBuffer(buffer);

    pos = pbr->getAttribLocation("roughness");
    buffer = GpuBuffer(GpuBufferType::PRIMITIVE_BUFFER, roughness.data(), roughness.size() * sizeof(float));
    buffer.EnableAttribute(pos, 1, GpuStorageType::FLOAT, false, 0, 0);
    buffers.AddBuffer(buffer);

    buffers.Bind();
}

void Renderer::_clearInstancedData(std::vector<GpuArrayBuffer> & gabuffers) {
    // glDeleteBuffers(buffers.size(), &buffers[0]);
    for (auto& buffer: gabuffers) buffer.Clear();
    gabuffers.clear();
}

void Renderer::_bindShader(Pipeline * s) {
    _unbindShader();
    s->bind();
    _state.currentShader = s;
}

void Renderer::_unbindShader() {
    if (!_state.currentShader) return;
    //_unbindAllTextures();
    _state.currentShader->unbind();
    _state.currentShader = nullptr;
}

static void setCullState(const RenderFaceCulling & mode) {
    // Set the culling state
    switch (mode) {
    case RenderFaceCulling::CULLING_NONE:
        glDisable(GL_CULL_FACE);
        break;
    case RenderFaceCulling::CULLING_CW:
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);
        glFrontFace(GL_CW);
        break;
    default:
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);
        glFrontFace(GL_CCW);
        break;
    }
}

void Renderer::_buildEntityList(const Camera & c) {
    for (auto& node : _state.entities) {
        const double distance = glm::distance(node.Get()->GetWorldPosition(), c.getPosition());
        if (distance < _state.zfar) {
            if (node.Get()->GetLightInteractionEnabled()) {
                addEntityMeshData(node, _state.instancedLightInteractMeshes);
            }
            else {
                addEntityMeshData(node, _state.instancedFlatMeshes);
            }
        }
    }
}

static bool ValidateTexture(const Async<Texture> & tex) {
    return tex.Completed() && !tex.Failed();
}

void Renderer::_render(const Camera & camera, const RenderNodeView& e, bool removeViewTranslation) {
    const glm::mat4 & projection = _state.perspective;
    //const glm::mat4 & view = c.getViewTransform();
    glm::mat4 view;
    if (removeViewTranslation) {
        // Remove the translation component of the view matrix
        view = glm::mat4(glm::mat3(camera.getViewTransform()));
    }
    else {
        view = camera.getViewTransform();
    }

    // Unbind current shader if one is bound
    _unbindShader();

    // Set up the shader we will use for this batch of entities
    Pipeline * s;
    std::vector<__MeshInstanceContainer>* meshContainer;
    if (e.Get()->GetLightInteractionEnabled() == false) {
        s = _state.forward.get();
        meshContainer = &_state.instancedFlatMeshes.find(e)->second;
    }
    else {
        s = _state.geometry.get();
        meshContainer = &_state.instancedLightInteractMeshes.find(e)->second;
    }

    //s->print();
    _bindShader(s);

    s->setMat4("projection", &projection[0][0]);
    s->setMat4("view", &view[0][0]);

#define SETUP_TEXTURE(name, flag, handle)           \
        tex = _lookupTexture(handle);               \
        const bool valid = ValidateTexture(tex);    \
        s->setBool(flag, valid);                    \
        if (valid) {                                \
            s->bindTexture(name, tex.Get());        \
        }

    for (int i = 0; i < meshContainer->size(); ++i) {

        Async<Texture> tex;
        const RenderMeshContainer* c = e.Get()->GetMeshContainer(i);
        const __MeshInstanceContainer& container = (*meshContainer)[i];

        if (c->material->GetDiffuseTexture()) {
            SETUP_TEXTURE("diffuseTexture", "textured", c->material->GetDiffuseTexture())
        }
        else {
            s->setBool("textured", false);
        }

        // Determine which uniforms we should set
        if (e.Get()->GetLightInteractionEnabled() == false) {
            s->setVec3("diffuseColor", &c->material->GetDiffuseColor()[0]);
        }
        else {
            if (c->material->GetNormalMap()) {
                SETUP_TEXTURE("normalMap", "normalMapped", c->material->GetNormalMap())
            }
            else {
                s->setBool("normalMapped", false);
            }

            if (c->material->GetDepthMap()) {
                //_bindTexture(s, "depthMap", m->getMaterial().depthMap);
                s->setFloat("heightScale", 0.01f);
                SETUP_TEXTURE("depthMap", "depthMapped", c->material->GetDepthMap())
            }
            else {
                s->setBool("depthMapped", false);
            }

            if (c->material->GetRoughnessMap()) {
                SETUP_TEXTURE("roughnessMap", "roughnessMapped", c->material->GetRoughnessMap());
            }
            else {
                s->setBool("roughnessMapped", false);
            }

            if (c->material->GetAmbientTexture()) {
                SETUP_TEXTURE("ambientOcclusionMap", "ambientMapped", c->material->GetAmbientTexture())
            }
            else {
                s->setBool("ambientMapped", false);
            }

            if (c->material->GetMetallicMap()) {
                SETUP_TEXTURE("metalnessMap", "metalnessMapped", c->material->GetMetallicMap())
            }
            else {
                s->setBool("metalnessMapped", false);
            }

            s->setVec3("viewPosition", &camera.getPosition()[0]);
        }

        // Perform instanced rendering
        setCullState(e.Get()->GetFaceCullMode());

        c->mesh->Render(container.size, container.buffers);
    }

#undef SETUP_TEXTURE

    _unbindShader();
}

void Renderer::_renderCSMDepth(const Camera & c, const std::unordered_map<RenderNodeView, std::vector<__MeshInstanceContainer>>& entities) {
    _bindShader(_state.csmDepth.get());
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    // Allows GPU to perform angle-dependent depth offset to help reduce artifacts such as shadow acne
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(0.5f, 1.0f);
    //glBlendFunc(GL_ONE, GL_ONE);
    // glDisable(GL_CULL_FACE);

    // Set up each individual view-projection matrix
    for (int i = 0; i < _state.csms.size(); ++i) {
        auto& csm = _state.csms[i];
        _state.csmDepth->setMat4("shadowMatrices[" + std::to_string(i) + "]", &csm.projectionView[0][0]);
    }

    // Render everything in a single pass
    _state.cascadeFbo.bind();
    const Texture * depth = _state.cascadeFbo.getDepthStencilAttachment();
    if (!depth) {
        throw std::runtime_error("Critical error: depth attachment not present");
    }
    glViewport(0, 0, depth->width(), depth->height());
    // Render each entity into the depth map
    for (auto& viewMesh : entities) {
        for (int i = 0; i < viewMesh.second.size(); ++i) {
            const __MeshInstanceContainer& container = viewMesh.second[i];
            const RenderNodeView& e = viewMesh.first;
            const RenderMeshPtr m = e.Get()->GetMeshContainer(i)->mesh;
            const size_t numInstances = container.size;
            setCullState(e.Get()->GetFaceCullMode());
            m->Render(numInstances, container.buffers);
        }
    }
    _state.cascadeFbo.unbind();

    _unbindShader();
    glDisable(GL_POLYGON_OFFSET_FILL);
}

void Renderer::end(const Camera & c) {
    for (auto& entityView : _entitiesSeenBefore) {
        auto rnode = entityView.first.Get()->GetRenderNode();
        bool anyIncomplete = false;
        for (int i = 0; i < rnode->GetNumMeshContainers(); ++i) {
            if (rnode->GetMeshContainer(i)->mesh->IsGpuDirty()) {
                anyIncomplete = true;
                rnode->GetMeshContainer(i)->mesh->GenerateGpuData();
            }
        }

        if (anyIncomplete) {
            ResourceManager::Instance()->FinalizeModelMemory(entityView.first.Get());
        }
    }

    _recalculateCascadeData(c);

    const int maxInstances = 250;
    const int maxShadowCastingLights = 8;
    const int maxTotalLights = 256;
    const int maxShadowUpdatesPerFrame = maxShadowCastingLights;
    // Need to delete these at the end of the frame
    std::vector<GpuArrayBuffer> buffers;

    //_unbindAllTextures();

    // We need to figure out what we want to attempt to render
    _buildEntityList(c);

    std::unordered_map<Light *, std::unordered_map<RenderNodeView, std::vector<__MeshInstanceContainer>>> perLightInstancedMeshes;
    std::unordered_map<Light *, bool> perLightIsDirty;
    std::vector<std::pair<Light *, double>> perLightDistToViewer;
    // This one is just for shadow-casting lights
    std::vector<std::pair<Light *, double>> perLightShadowCastingDistToViewer;
    // Init per light instance data
    for (Light * light : _state.lights) {
        const double distance = glm::distance(c.getPosition(), light->position);
        perLightDistToViewer.push_back(std::make_pair(light, distance));
        //if (distance > 2 * light->getRadius()) continue;
        perLightInstancedMeshes.insert(std::make_pair(light, std::unordered_map<RenderNodeView, std::vector<__MeshInstanceContainer>>()));
        perLightIsDirty.insert(std::make_pair(light, _lightsSeenBefore.find(light)->second.dirty));
        if (light->castsShadows()) {
            perLightShadowCastingDistToViewer.push_back(std::make_pair(light, distance));
        }
    }

    // Sort lights based on distance to viewer
    const auto comparison = [](const std::pair<Light *, double> & a, const std::pair<Light *, double> & b) {
        return a.second < b.second;
    };
    std::sort(perLightDistToViewer.begin(), perLightDistToViewer.end(), comparison);
    std::sort(perLightShadowCastingDistToViewer.begin(), perLightShadowCastingDistToViewer.end(), comparison);

    // Remove lights exceeding the absolute maximum
    if (perLightDistToViewer.size() > maxTotalLights) {
        perLightDistToViewer.resize(maxTotalLights);
    }

    // Remove shadow-casting lights that exceed our max count
    if (perLightShadowCastingDistToViewer.size() > maxShadowCastingLights) {
        perLightShadowCastingDistToViewer.resize(maxShadowCastingLights);
    }

    for (const RenderNodeView& e : _state.lightInteractingEntities) {
        const bool entityIsDirty = _entitiesSeenBefore.find(EntityView(e.Get()->GetOwner()))->second.dirty;
        for (auto&[light, _] : perLightShadowCastingDistToViewer) {
            const double distance = glm::distance(e.Get()->GetWorldPosition(), light->position);
            if (distance > light->getRadius()) continue;
            addEntityMeshData(e, perLightInstancedMeshes.find(light)->second);
            perLightIsDirty.find(light)->second |= entityIsDirty;
        }
    }

    // Set blend func just for shadow pass
    // glBlendFunc(GL_ONE, GL_ONE);
    glEnable(GL_DEPTH_TEST);
    // Perform the shadow volume pre-pass
    _bindShader(_state.shadows.get());
    int shadowUpdates = 0;
    for (auto&[light, d] : perLightShadowCastingDistToViewer) {
        if (shadowUpdates > maxShadowUpdatesPerFrame) break;
        ++shadowUpdates;
        const double distance = glm::distance(c.getPosition(), light->position);
        // We want to compute shadows at least once for each light source before we enable the option of skipping it 
        // due to it being too far away
        const bool dirty = _lightsSeenBefore.find(light)->second.dirty || perLightIsDirty.find(light)->second;
        //if (distance > 2 * light->getRadius() || !dirty) continue;
        if (!dirty) continue;

        // Set dirty to false
        _lightsSeenBefore.find(light)->second.dirty = false;

        auto & instancedMeshes = perLightInstancedMeshes.find(light)->second;

        // Init the instance data which enables us to drastically reduce the number of draw calls
        for (auto& viewMesh : instancedMeshes) {
            for (auto& mesh : viewMesh.second) {
                _initInstancedData(mesh, buffers);
            }
        }
    
        // TODO: Make this work with spotlights
        PointLight * point = (PointLight *)light;
        const ShadowMap3D & smap = this->_shadowMap3DHandles.find(_getShadowMapHandleForLight(point))->second;

        const glm::mat4 lightPerspective = glm::perspective<float>(glm::radians(90.0f), float(smap.shadowCubeMap.width()) / smap.shadowCubeMap.height(), point->getNearPlane(), point->getFarPlane());

        // glBindFramebuffer(GL_FRAMEBUFFER, smap.frameBuffer);
        smap.frameBuffer.clear(glm::vec4(1.0f));
        smap.frameBuffer.bind();
        glViewport(0, 0, smap.shadowCubeMap.width(), smap.shadowCubeMap.height());
        // Current pass only cares about depth buffer
        // glClear(GL_DEPTH_BUFFER_BIT);

        auto transforms = generateLightViewTransforms(lightPerspective, point->position);
        for (int i = 0; i < transforms.size(); ++i) {
            const std::string index = "[" + std::to_string(i) + "]";
            _state.shadows->setMat4("shadowMatrices" + index, &transforms[i][0][0]);
        }
        _state.shadows->setVec3("lightPos", &light->position[0]);
        _state.shadows->setFloat("farPlane", point->getFarPlane());

        for (auto & entityObservers : instancedMeshes) {
            for (int i = 0; i < entityObservers.second.size(); ++i) {
                RenderMeshPtr m = entityObservers.first.Get()->GetMeshContainer(i)->mesh;
                setCullState(entityObservers.first.Get()->GetFaceCullMode());
                m->Render(entityObservers.second[i].size, entityObservers.second[i].buffers);
            }
        }

        // Unbind
        //glBindFramebuffer(GL_FRAMEBUFFER, 0);
        smap.frameBuffer.unbind();
        _clearInstancedData(buffers);
    }
    _clearInstancedData(buffers);
    //_unbindAllTextures();
    _unbindShader();

    // Init the instance data which enables us to drastically reduce the number of draw calls
    for (auto & entityObservers : _state.instancedLightInteractMeshes) {
        for (auto & meshObservers : entityObservers.second) {
            _initInstancedData(meshObservers, buffers);
        }
    }

    for (auto& entityObservers : _state.instancedFlatMeshes) {
        for (auto& meshObservers : entityObservers.second) {
            _initInstancedData(meshObservers, buffers);
        }
    }

    // Perform world light depth pass if enabled
    if (_state.worldLightingEnabled) {
        _renderCSMDepth(c, _state.instancedLightInteractMeshes);
    }

    // TEMP: Set up the light source
    //glm::vec3 lightPos(0.0f, 0.0f, 0.0f);
    //glm::vec3 lightColor(10.0f); 

    // Make sure to bind our own frame buffer for rendering
    _state.buffer.fbo.bind();
    
    // Make sure some of our global GL states are set properly for primary rendering below
    glBlendFunc(_state.blendSFactor, _state.blendDFactor);
    glViewport(0, 0, _state.windowWidth, _state.windowHeight);

    // Begin geometry pass
    //glEnable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    for (auto & entityObservers : _state.instancedLightInteractMeshes) {
        const RenderNodeView& e = entityObservers.first;
        _render(c, e);
    }
    _state.buffer.fbo.unbind();

    glDisable(GL_CULL_FACE);
    //glEnable(GL_BLEND);

    // Begin deferred lighting pass
    _state.lightingFbo.bind();
    glDisable(GL_DEPTH_TEST);
    //_unbindAllTextures();
    _bindShader(_state.lighting.get());
    _initLights(_state.lighting.get(), c, perLightDistToViewer, maxShadowCastingLights);
    _state.lighting->bindTexture("gPosition", _state.buffer.position);
    _state.lighting->bindTexture("gNormal", _state.buffer.normals);
    _state.lighting->bindTexture("gAlbedo", _state.buffer.albedo);
    _state.lighting->bindTexture("gBaseReflectivity", _state.buffer.baseReflectivity);
    _state.lighting->bindTexture("gRoughnessMetallicAmbient", _state.buffer.roughnessMetallicAmbient);
    _renderQuad();
    _state.lightingFbo.unbind();
    _unbindShader();

    // Forward pass for all objects that don't interact with light (may also be used for transparency later as well)
    _state.lightingFbo.copyFrom(_state.buffer.fbo, BufferBounds{0, 0, _state.windowWidth, _state.windowHeight}, BufferBounds{0, 0, _state.windowWidth, _state.windowHeight}, BufferBit::DEPTH_BIT, BufferFilter::NEAREST);
    // Blit to default framebuffer - not that the framebuffer you are writing to has to match the internal format
    // of the framebuffer you are reading to!
    glEnable(GL_DEPTH_TEST);
    _state.lightingFbo.bind();
    for (auto & entityObservers : _state.instancedFlatMeshes) {
        const RenderNodeView& e = entityObservers.first;
        _render(c, e);
    }
    _state.lightingFbo.unbind();
    // glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Enable post-FX effects such as bloom
    _performPostFxProcessing();

    // Perform final drawing to screen + gamma correction
    _finalizeFrame();

    // Make sure to clear out all instanced data used this frame
    _clearInstancedData(buffers);
}

void Renderer::_performPostFxProcessing() {
    glDisable(GL_CULL_FACE);
    glDisable(GL_BLEND);
    //glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);

    // We use this so that we can avoid a final copy between the downsample and blurring stages
    std::vector<PostFXBuffer> finalizedPostFxFrames(_state.numDownsampleIterations + _state.numUpsampleIterations);
   
    Pipeline* bloom = _state.bloom.get();
    _bindShader(bloom);

    // Downsample stage
    bloom->setBool("downsamplingStage", true);
    bloom->setBool("upsamplingStage", false);
    bloom->setBool("finalStage", false);
    bloom->setBool("gaussianStage", false);
    for (int i = 0, gaussian = 0; i < _state.numDownsampleIterations; ++i, gaussian += 2) {
        PostFXBuffer& buffer = _state.postFxBuffers[i];
        Texture colorTex = buffer.fbo.getColorAttachments()[0];
        auto width = colorTex.width();
        auto height = colorTex.height();
        bloom->setFloat("viewportX", float(width));
        bloom->setFloat("viewportY", float(height));
        buffer.fbo.bind();
        glViewport(0, 0, width, height);
        if (i == 0) {
            bloom->bindTexture("mainTexture", _state.lightingColorBuffer);
        }
        else {
            bloom->bindTexture("mainTexture", _state.postFxBuffers[i - 1].fbo.getColorAttachments()[0]);
        }
        _renderQuad();
        buffer.fbo.unbind();

        // Now apply Gaussian blurring
        bool horizontal = false;
        bloom->setBool("downsamplingStage", false);
        bloom->setBool("gaussianStage", true);
        BufferBounds bounds = BufferBounds{0, 0, width, height};
        for (int i = 0; i < 2; ++i) {
            FrameBuffer& blurFbo = _state.gaussianBuffers[gaussian + i].fbo;
            FrameBuffer copyFromFbo;
            if (i == 0) {
                copyFromFbo = buffer.fbo;
            }
            else {
                copyFromFbo = _state.gaussianBuffers[gaussian].fbo;
            }

            bloom->setBool("horizontal", horizontal);
            bloom->bindTexture("mainTexture", copyFromFbo.getColorAttachments()[0]);
            horizontal = !horizontal;
            blurFbo.bind();
            _renderQuad();
            blurFbo.unbind();
        }

        // Copy the end result back to the original buffer
        // buffer.fbo.copyFrom(_state.gaussianBuffers[gaussian + 1].fbo, bounds, bounds, BufferBit::COLOR_BIT, BufferFilter::LINEAR);
        finalizedPostFxFrames[i] = _state.gaussianBuffers[gaussian + 1];
    }

    // Upsample stage
    bloom->setBool("downsamplingStage", false);
    bloom->setBool("upsamplingStage", true);
    bloom->setBool("finalStage", false);
    bloom->setBool("gaussianStage", false);
    int postFXIndex = _state.numDownsampleIterations;
    for (int i = _state.numDownsampleIterations - 1; i >= 0; --i, ++postFXIndex) {
        PostFXBuffer& buffer = _state.postFxBuffers[postFXIndex];
        auto width = buffer.fbo.getColorAttachments()[0].width();
        auto height = buffer.fbo.getColorAttachments()[0].height();
        bloom->setFloat("viewportX", float(width));
        bloom->setFloat("viewportY", float(height));
        buffer.fbo.bind();
        glViewport(0, 0, width, height);
        //bloom->bindTexture("mainTexture", _state.postFxBuffers[postFXIndex - 1].fbo.getColorAttachments()[0]);
        bloom->bindTexture("mainTexture", finalizedPostFxFrames[postFXIndex - 1].fbo.getColorAttachments()[0]);
        if (i == 0) {
            bloom->bindTexture("bloomTexture", _state.lightingColorBuffer);
            bloom->setBool("finalStage", true);
        }
        else {
            //bloom->bindTexture("bloomTexture", _state.postFxBuffers[i - 1].fbo.getColorAttachments()[0]);
            bloom->bindTexture("bloomTexture", finalizedPostFxFrames[i - 1].fbo.getColorAttachments()[0]);
        }
        _renderQuad();
        buffer.fbo.unbind();
        
        finalizedPostFxFrames[postFXIndex] = buffer;
        _state.finalScreenTexture = buffer.fbo.getColorAttachments()[0];
    }

    _unbindShader();
}

void Renderer::_finalizeFrame() {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDisable(GL_CULL_FACE);
    glViewport(0, 0, _state.windowWidth, _state.windowHeight);
    //glEnable(GL_BLEND);

    // Now render the screen
    _bindShader(_state.hdrGamma.get());
    _state.hdrGamma->bindTexture("screen", _state.finalScreenTexture);
    _renderQuad();
    _unbindShader();
}

void Renderer::_renderQuad() {
    _state.screenQuad->GetMeshContainer(0)->mesh->Render(1, GpuArrayBuffer());
}

TextureHandle Renderer::createShadowMap3D(uint32_t resolutionX, uint32_t resolutionY) {
    ShadowMap3D smap;
    smap.shadowCubeMap = Texture(TextureConfig{TextureType::TEXTURE_3D, TextureComponentFormat::DEPTH, TextureComponentSize::BITS_DEFAULT, TextureComponentType::FLOAT, resolutionX, resolutionY, 0, false}, nullptr);
    smap.shadowCubeMap.setMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);
    smap.shadowCubeMap.setCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);

    smap.frameBuffer = FrameBuffer({smap.shadowCubeMap});
    if (!smap.frameBuffer.valid()) {
        _isValid = false;
        return TextureHandle::Null();
    }
    TextureHandle handle = TextureHandle::NextHandle();
    this->_shadowMap3DHandles.insert(std::make_pair(handle, smap));
    return handle;
}

Async<Texture> Renderer::_lookupTexture(TextureHandle handle) const {
    Async<Texture> ret;
    ResourceManager::Instance()->GetTexture(handle, ret);
    return ret;
}

Texture Renderer::_lookupShadowmapTexture(TextureHandle handle) const {
    if (handle == TextureHandle::Null()) return Texture();

    if (_shadowMap3DHandles.find(handle) == _shadowMap3DHandles.end()) {
        return Texture();
    }

    return _shadowMap3DHandles.find(handle)->second.shadowCubeMap;
}

// TODO: Need a way to clean up point light resources
void Renderer::addPointLight(Light * light) {
    assert(light->getType() == LightType::POINTLIGHT || light->getType() == LightType::SPOTLIGHT);
    _state.lights.push_back(light);

    // if (light->getType() == LightType::POINTLIGHT) {
    //     PointLight * point = (PointLight *)light;
    //     if (_getShadowMapHandleForLight(light) == -1) {
    //         _setLightShadowMapHandle(light, this->createShadowMap3D(_state.shadowCubeMapX, _state.shadowCubeMapY));
    //     }
    // }

    if (_lightsSeenBefore.find(light) == _lightsSeenBefore.end()) {
        _lightsSeenBefore.insert(std::make_pair(light, EntityStateInfo{light->position, glm::vec3(0.0f), glm::vec3(0.0f), true}));
    }
    else {
        EntityStateInfo & info = _lightsSeenBefore.find(light)->second;
        // If no associated shadow map, mark as dirty
        if (_lightsToShadowMap.find(light) == _lightsToShadowMap.end()) {
            info.dirty = true;
        }
        const double distance = glm::distance(light->position, info.lastPos);
        if (distance > 0.25) {
            info.lastPos = light->position;
            info.dirty = true;
        }
        //else {
        //    info.dirty = false;
        //}
    }
}

void Renderer::_initLights(Pipeline * s, const Camera & c, const std::vector<std::pair<Light *, double>> & lights, const size_t maxShadowLights) {
    // Set up point lights

    glm::vec3 lightColor;
    int lightIndex = 0;
    int shadowLightIndex = 0;
    int i = 0;
    for (; i < lights.size(); ++i) {
        PointLight * light = (PointLight *)lights[i].first;
        const double distance = lights[i].second; //glm::distance(c.getPosition(), light->position);
        // Skip lights too far from camera
        //if (distance > (2 * light->getRadius())) continue;
        lightColor = light->getBaseColor() * light->getIntensity();
        s->setVec3("lightPositions[" + std::to_string(lightIndex) + "]", &light->position[0]);
        s->setVec3("lightColors[" + std::to_string(lightIndex) + "]", &lightColor[0]);
        s->setFloat("lightRadii[" + std::to_string(lightIndex) + "]", light->getRadius());
        s->setBool("lightCastsShadows[" + std::to_string(lightIndex) + "]", light->castsShadows());
        //_bindShadowMapTexture(s, "shadowCubeMaps[" + std::to_string(lightIndex) + "]", light->getShadowMapHandle());
        if (light->castsShadows() && shadowLightIndex < maxShadowLights) {
            s->setFloat("lightFarPlanes[" + std::to_string(shadowLightIndex) + "]", light->getFarPlane());
            //_bindShadowMapTexture(s, "shadowCubeMaps[" + std::to_string(shadowLightIndex) + "]", _getShadowMapHandleForLight(light));
            s->bindTexture("shadowCubeMaps[" + std::to_string(shadowLightIndex) + "]", _lookupShadowmapTexture(_getShadowMapHandleForLight(light)));
            ++shadowLightIndex;
        }
        ++lightIndex;
    }

    if (shadowLightIndex == 0) {
       // If we don't do this the fragment shader crashes
       s->setFloat("lightFarPlanes[0]", 0.0f);
       //_bindShadowMapTexture(s, "shadowCubeMaps[0]", _state.dummyCubeMap);
       s->bindTexture("shadowCubeMaps[0]", _lookupShadowmapTexture(_state.dummyCubeMap));
    }

    s->setFloat("ambientIntensity", 0.0001f);
    /*
    if (lightIndex == 0) {
        s->setFloat("ambientIntensity", 0.0001f);
    }
    else {
        s->setFloat("ambientIntensity", 0.0f);
    }
    */

    s->setInt("numLights", lightIndex);
    s->setInt("numShadowLights", shadowLightIndex);
    s->setVec3("viewPosition", &c.getPosition()[0]);

    // Set up world light if enabled
    //glm::mat4 lightView = constructViewMatrix(_state.worldLight.getRotation(), _state.worldLight.getPosition());
    //glm::mat4 lightView = constructViewMatrix(_state.worldLight.getRotation(), glm::vec3(0.0f));
    // Camera lightCam(false);
    // lightCam.setAngle(_state.worldLight.getRotation());
    const Camera & lightCam = _state.worldLightCamera;
    glm::mat4 lightWorld = lightCam.getWorldTransform();
    // glm::mat4 lightView = lightCam.getViewTransform();
    glm::vec3 direction = lightCam.getDirection(); //glm::vec3(-lightWorld[2].x, -lightWorld[2].y, -lightWorld[2].z);
    // STRATUS_LOG << "Light direction: " << direction << std::endl;
    s->setBool("infiniteLightingEnabled", _state.worldLightingEnabled);
    s->setVec3("infiniteLightDirection", &direction[0]);
    lightColor = _state.worldLight.getColor() * _state.worldLight.getIntensity();
    s->setVec3("infiniteLightColor", &lightColor[0]);

    s->bindTexture("infiniteLightShadowMap", *_state.cascadeFbo.getDepthStencilAttachment());
    for (int i = 0; i < 4; ++i) {
        //s->bindTexture("infiniteLightShadowMaps[" + std::to_string(i) + "]", *_state.csms[i].fbo.getDepthStencilAttachment());
        s->setMat4("cascadeProjViews[" + std::to_string(i) + "]", &_state.csms[i].projectionView[0][0]);
        // s->setFloat("cascadeSplits[" + std::to_string(i) + "]", _state.cascadeSplits[i]);
    }

    for (int i = 0; i < 2; ++i) {
        s->setVec4("shadowOffset[" + std::to_string(i) + "]", &_state.cascadeShadowOffsets[i][0]);
    }

    for (int i = 0; i < 3; ++i) {
        // s->setVec3("cascadeScale[" + std::to_string(i) + "]", &_state.csms[i + 1].cascadeScale[0]);
        // s->setVec3("cascadeOffset[" + std::to_string(i) + "]", &_state.csms[i + 1].cascadeOffset[0]);
        s->setVec4("cascadePlanes[" + std::to_string(i) + "]", &_state.csms[i + 1].cascadePlane[0]);
    }

    // s->setMat4("cascade0ProjView", &_state.csms[0].projectionView[0][0]);
}

TextureHandle Renderer::_getShadowMapHandleForLight(Light * light) {
    assert(_shadowMap3DHandles.size() > 0);

    auto it = _lightsToShadowMap.find(light);
    // If not found, look for an existing shadow map
    if (it == _lightsToShadowMap.end()) {
        // Mark the light as dirty since its map will need to be updated
        _lightsSeenBefore.find(light)->second.dirty = true;

        TextureHandle handle;
        for (const auto & entry : _shadowMap3DHandles) {
            if (_usedShadowMaps.find(entry.first) == _usedShadowMaps.end()) {
                handle = entry.first;
                break;
            }
        }

        if (handle == TextureHandle::Null()) {
            // Evict oldest since we could not find an available handle
            Light * oldest = _lruLightCache.front();
            _lruLightCache.pop_front();
            handle = _lightsToShadowMap.find(oldest)->second;
            _evictLightFromShadowMapCache(oldest);
        }

        _setLightShadowMapHandle(light, handle);
        _addLightToShadowMapCache(light);
        return handle;
    }

    // Update the LRU cache
    _addLightToShadowMapCache(light);
    return it->second;
}

void Renderer::_setLightShadowMapHandle(Light * light, TextureHandle handle) {
    _lightsToShadowMap.insert(std::make_pair(light, handle));
    _usedShadowMaps.insert(handle);
}

void Renderer::_evictLightFromShadowMapCache(Light * light) {
    for (auto it = _lruLightCache.begin(); it != _lruLightCache.end(); ++it) {
        if (*it == light) {
            _lruLightCache.erase(it);
            return;
        }
    }
}

void Renderer::_addLightToShadowMapCache(Light * light) {
    // First remove the existing light entry if it's already there
    _evictLightFromShadowMapCache(light);
    // Push to back so that it is seen as most recently used
    _lruLightCache.push_back(light);
}

// Allows user to toggle on/off the global infinite light
void Renderer::toggleWorldLighting(bool enabled) {
    _state.worldLightingEnabled = enabled;
}

// Allows user to modify world light properties
void Renderer::setWorldLight(const InfiniteLight & wl) {
    _state.worldLight = wl;
    _state.worldLightIsDirty = true;
    _state.worldLightCamera = Camera(false);
    _state.worldLightCamera.setAngle(_state.worldLight.getRotation());
}
}