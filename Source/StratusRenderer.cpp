
#include <StratusRenderer.h>
#include <iostream>
#include <StratusLight.h>
#include "StratusPipeline.h"
#include "StratusRenderer.h"
#include <math.h>
#include <cmath>
#include "StratusQuad.h"
#include "StratusCube.h"
#include "StratusUtils.h"
#include "StratusMath.h"
#define STB_IMAGE_IMPLEMENTATION
#include "STBImage.h"

namespace stratus {
bool __RenderEntityObserver::operator==(const __RenderEntityObserver & c) const {
    return (*e) == *(c.e);
}

size_t __RenderEntityObserver::hashCode() const {
    return e->hashCode();
}

bool __MeshObserver::operator==(const __MeshObserver & c) const {
    return (*m) == *(c.m);
}

size_t __MeshObserver::hashCode() const {
    return m->hashCode();
}

static void printGLInfo(const GFXConfig & config) {
    std::cout << "==================== OpenGL Information ====================" << std::endl;
    std::cout << "\tRenderer: "                         << config.renderer << std::endl;
    std::cout << "\tVersion: "                          << config.version << std::endl;
    std::cout << "\tMax draw buffers: "                 << config.maxDrawBuffers << std::endl;
    std::cout << "\tMax combined textures: "            << config.maxCombinedTextures << std::endl;
    std::cout << "\tMax cube map texture size: "        << config.maxCubeMapTextureSize << std::endl;
    std::cout << "\tMax fragment uniform vectors: "     << config.maxFragmentUniformVectors << std::endl;
    std::cout << "\tMax fragment uniform components: "  << config.maxFragmentUniformComponents << std::endl;
    std::cout << "\tMax varying floats: "               << config.maxVaryingFloats << std::endl;
    std::cout << "\tMax render buffer size: "           << config.maxRenderbufferSize << std::endl;
    std::cout << "\tMax texture image units: "          << config.maxTextureImageUnits << std::endl;
    std::cout << "\tMax texture size: "                 << config.maxTextureSize << std::endl;
    std::cout << "\tMax vertex attribs: "               << config.maxVertexAttribs << std::endl;
    std::cout << "\tMax vertex uniform vectors: "       << config.maxVertexUniformVectors << std::endl;
    std::cout << "\tMax vertex uniform components: "    << config.maxVertexUniformComponents << std::endl;
    std::cout << "\tMax viewport dims: "                << "(" << config.maxViewportDims[0] << ", " << config.maxViewportDims[1] << ")" << std::endl;
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
        std::cerr << "[error] Unable to create a valid OpenGL context" << std::endl;
        _isValid = false;
        return;
    }

    // Init gl core profile using gl3w
    if (gl3wInit()) {
        std::cerr << "[error] Failed to initialize core OpenGL profile" << std::endl;
        _isValid = false;
        return;
    }

    //if (!gl3wIsSupported(maxGLVersion, minGLVersion)) {
    //    std::cerr << "[error] OpenGL 3.2 not supported" << std::endl;
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
    
    // Now we need to establish a mapping between all of the possible render
    // property combinations with a list of entities that match those requirements
    _state.entities.insert(make_pair(FLAT, vector<RenderEntity *>()));
    _state.entities.insert(make_pair(DYNAMIC, vector<RenderEntity *>()));

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
        Shader{"../resources/shaders/csm.fs", ShaderType::FRAGMENT}}));
    _state.shaders.push_back(_state.csmDepth.get());

    // Create the screen quad
    _state.screenQuad = std::make_unique<Quad>();

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
    invalidateAllTextures();

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
    static constexpr int cascadeResolutionXY = 2048;
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
    _state.cascadeShadowOffsets[0] = glm::vec4(-cascadeDelta, -2.0f * cascadeDelta, 2.0f * cascadeDelta, -cascadeDelta);
    _state.cascadeShadowOffsets[1] = glm::vec4(cascadeDelta, 2.0f * cascadeDelta, -2.0f * cascadeDelta, cascadeDelta);
    // _state.cascadeShadowOffsets[0] = glm::vec4(-cascadeDelta, -cascadeDelta, cascadeDelta, -cascadeDelta);
    // _state.cascadeShadowOffsets[1] = glm::vec4(cascadeDelta, cascadeDelta, -cascadeDelta, cascadeDelta);

    // Assume directional light translation is none
    Camera light(false);
    light.setAngle(_state.worldLight.getRotation());
    const glm::mat4& lightWorldTransform = light.getWorldTransform();
    const glm::mat4& lightViewTransform = light.getViewTransform();
    const glm::mat4& cameraWorldTransform = c.getWorldTransform();
    const glm::mat4 transposeLightWorldTransform = glm::transpose(lightWorldTransform);

    const glm::mat4 L = lightViewTransform * cameraWorldTransform;

    const float s = float(_state.windowWidth) / float(_state.windowHeight);
    const float g = 1.0 / tangent(_state.fov / 2.0f).value();
    //const float tanHalfFovVertical = std::tanf(glm::radians((_state.fov * s) / 2.0f));
    // std::cout << "AAAAAAA " << g << ", " << _state.fov << ", " << _state.fov / 2.0f << std::endl;
    const float znear = _state.znear;
    const float zfar = _state.zfar;

    // @see https://johanmedestrom.wordpress.com/2016/03/18/opengl-cascaded-shadow-maps/
    // @see https://johanmedestrom.wordpress.com/2016/03/18/opengl-cascaded-shadow-maps/
    const float lambda = 0.5f;
    const float clipRange = zfar - znear;
    const float ratio = zfar / znear;
    std::vector<float> cascadeEnds(numCascades);
    for (int i = 0; i < numCascades; ++i) {
        const float p = (i + 1) / float(numCascades);
        const float log = znear * std::pow(ratio, p);
        const float uniform = znear + clipRange * p;
        const float d = std::floorf(lambda * (log - uniform) + uniform);
        cascadeEnds[i] = d;
    }

    const std::vector<float> cascadeBegins = { 0.0f, cascadeEnds[0] - 20.0f,  cascadeEnds[1] - 20.0f, cascadeEnds[2] - 20.0f }; // 4 cascades max
    //const std::vector<float> cascadeEnds   = {  30.0f, 100.0f, 240.0f, 640.0f };
    std::vector<float> aks;
    std::vector<float> bks;
    std::vector<float> dks;
    std::vector<glm::vec3> sks;
    std::vector<float> zmins;
    std::vector<float> zmaxs;

    for (int i = 0; i < numCascades; ++i) {
        if (recalculateFbos) {
            // Create the depth buffer
            // @see https://stackoverflow.com/questions/22419682/glsl-sampler2dshadow-and-shadow2d-clarificationssss
            Texture tex(TextureConfig{ TextureType::TEXTURE_2D, TextureComponentFormat::DEPTH, TextureComponentSize::BITS_DEFAULT, TextureComponentType::FLOAT, cascadeResolutionXY, cascadeResolutionXY, false }, nullptr);
            tex.setMinMagFilter(TextureMinificationFilter::NEAREST, TextureMagnificationFilter::NEAREST);
            tex.setCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);
            tex.setTextureCompare(TextureCompareMode::COMPARE_REF_TO_TEXTURE, TextureCompareFunc::LEQUAL);

            // Create the frame buffer
            _state.csms[i].fbo = FrameBuffer({ tex });
        }
    
        const float ak = cascadeBegins[i];
        const float bk = cascadeEnds[i];
        aks.push_back(ak);
        bks.push_back(bk);

        // These base values are in camera space
        const float baseAkX = (ak * s) / g;
        const float baseAkY = ak / g;
        const float baseBkX = (bk * s) / g;
        const float baseBkY = bk / g;
        // Keep all of these in camera space for now
        std::vector<glm::vec4> frustumCorners = {
            // Near corners
            glm::vec4(baseAkX, -baseAkY, ak, 1.0f),
            glm::vec4(baseAkX, baseAkY, ak, 1.0f),
            glm::vec4(-baseAkX, baseAkY, ak, 1.0f),
            glm::vec4(-baseAkX, -baseAkY, ak, 1.0f),

            // Far corners
            glm::vec4(baseBkX, -baseBkY, bk, 1.0f),
            glm::vec4(baseBkX, baseBkY, bk, 1.0f),
            glm::vec4(-baseBkX, baseBkY, bk, 1.0f),
            glm::vec4(-baseBkX, -baseBkY, bk, 1.0f),
        };
        
        // This tells us the maximum diameter for the cascade bounding box k
        const float dk = std::ceilf(std::max<float>(glm::length(frustumCorners[0] - frustumCorners[6]), 
                                                    glm::length(frustumCorners[4] - frustumCorners[6])));
        dks.push_back(dk);
        const float T = dk / cascadeResolutionXY;

        // Compute min/max of each
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
        // std::cout << "sk " << sk << std::endl;
        sks.push_back(sk);

        // We use transposeLightWorldTransform because it's less precision-error-prone than just doing glm::inverse(lightWorldTransform)
        // Note: we use -sk instead of lightWorldTransform * sk because we're assuming the translation component is 0
        // glm::mat4 cascadeViewTransform = glm::mat4(1.0f);
        // cascadeViewTransform[3] = glm::vec4(-sk, 1.0f);
        // cascadeViewTransform = lightViewTransform * cascadeViewTransform;
        const glm::mat4 cascadeViewTransform = glm::mat4(transposeLightWorldTransform[0], 
                                                         transposeLightWorldTransform[1],
                                                         transposeLightWorldTransform[2],
                                                         glm::vec4(-sk, 1.0f));
        //const glm::mat4 cascadeViewTransform = lightViewTransform;

        // We are putting the light camera location sk on the near plane in the halfway point between left, right, top and bottom planes
        // so it enables us to use the simplified Orthographic Projection matrix below
        //
        // This results in values between [-1, 1]
        const glm::mat4 cascadeOrthoProjection(glm::vec4(2.0f / dk, 0.0f, 0.0f, 0.0f), 
                                               glm::vec4(0.0f, 2.0f / dk, 0.0f, 0.0f),
                                               glm::vec4(0.0f, 0.0f, 1.0f / (maxZ - minZ), 0.0f),
                                               glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));

        const glm::mat4 cascadeTexelOrthoProjection(glm::vec4(1.0f / dk, 0.0f, 0.0f, 0.0f), 
                                                    glm::vec4(0.0f, 1.0f / dk, 0.0f, 0.0f),
                                                    glm::vec4(0.0f, 0.0f, 1.0f / (maxZ - minZ), 0.0f),
                                                    glm::vec4(0.5f, 0.5f, 0.0f, 1.0f));

        _state.csms[i].depthProjection = cascadeOrthoProjection;
        _state.csms[i].texelProjection = cascadeTexelOrthoProjection;
        _state.csms[i].view = cascadeViewTransform;
        _state.csms[i].projectionView = cascadeTexelOrthoProjection * cascadeViewTransform;

        // std::cout << -glm::inverse(cascadeViewTransform)[2] << std::endl;
        // std::cout << light.getDirection() << std::endl;

        // std::cout << cascadeOrthoProjection << std::endl;
        // std::cout << glm::ortho(minX, maxX, minY, maxY, minZ, maxZ) << std::endl;

        if (i > 0) {
            // This will allow us to calculate the cascade blending weights in the vertex shader and then
            // the cascade indices in the pixel shader
            const glm::vec3 n = -glm::vec3(cameraWorldTransform[2]);
            const glm::vec3 c = glm::vec3(cameraWorldTransform[3]);
            const glm::vec4 fk = glm::vec4(n.x, n.y, n.z, glm::dot(-n, c) - ak) * (1.0f / (bks[i - 1] - ak));
            _state.csms[i].cascadePlane = fk;
            // We need an easy way to transform a texture coordinate in cascade 0 to any of the other cascades, which is
            // what cascadeScale and cascadeOffset allow us to do
            //
            // (These are actually pulled from a matrix which corresponds to PkShadow * MkView)
            _state.csms[i].cascadeScale = glm::vec3(dks[0] / dk, 
                                                    dks[0] / dk, 
                                                    (zmaxs[0] - zmins[0]) / (zmaxs[i] - zmins[i]));
            const float d02dk = dks[0] / (2.0f * dk);
            _state.csms[i].cascadeOffset = glm::vec3(((sks[0].x - sk.x) / dk) - d02dk + 0.5f, 
                                                     ((sks[0].y - sk.y) / dk) - d02dk + 0.5f, 
                                                     (sks[0].z - sk.z) / (maxZ - minZ));
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
    buffer.position = Texture(TextureConfig{TextureType::TEXTURE_2D, TextureComponentFormat::RGB, TextureComponentSize::BITS_32, TextureComponentType::FLOAT, _state.windowWidth, _state.windowHeight, false}, nullptr);
    buffer.position.setMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);

    // Normal buffer
    buffer.normals = Texture(TextureConfig{TextureType::TEXTURE_2D, TextureComponentFormat::RGB, TextureComponentSize::BITS_32, TextureComponentType::FLOAT, _state.windowWidth, _state.windowHeight, false}, nullptr);
    buffer.normals.setMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);

    // Create the color buffer - notice that is uses higher
    // than normal precision. This allows us to write color values
    // greater than 1.0 to support things like HDR.
    buffer.albedo = Texture(TextureConfig{TextureType::TEXTURE_2D, TextureComponentFormat::RGB, TextureComponentSize::BITS_16, TextureComponentType::FLOAT, _state.windowWidth, _state.windowHeight, false}, nullptr);
    buffer.albedo.setMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);

    // Base reflectivity buffer
    buffer.baseReflectivity = Texture(TextureConfig{TextureType::TEXTURE_2D, TextureComponentFormat::RGB, TextureComponentSize::BITS_16, TextureComponentType::FLOAT, _state.windowWidth, _state.windowHeight, false}, nullptr);
    buffer.baseReflectivity.setMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);

    // Roughness-Metallic-Ambient buffer
    buffer.roughnessMetallicAmbient = Texture(TextureConfig{TextureType::TEXTURE_2D, TextureComponentFormat::RGB, TextureComponentSize::BITS_16, TextureComponentType::FLOAT, _state.windowWidth, _state.windowHeight, false}, nullptr);
    buffer.roughnessMetallicAmbient.setMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);

    // Create the depth buffer
    buffer.depth = Texture(TextureConfig{TextureType::TEXTURE_2D, TextureComponentFormat::DEPTH, TextureComponentSize::BITS_DEFAULT, TextureComponentType::FLOAT, _state.windowWidth, _state.windowHeight, false}, nullptr);
    buffer.depth.setMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);

    // Create the frame buffer with all its texture attachments
    buffer.fbo = FrameBuffer({buffer.position, buffer.normals, buffer.albedo, buffer.baseReflectivity, buffer.roughnessMetallicAmbient, buffer.depth});
    if (!buffer.fbo.valid()) {
        _isValid = false;
        return;
    }

    // Code to create the lighting fbo
    _state.lightingColorBuffer = Texture(TextureConfig{TextureType::TEXTURE_2D, TextureComponentFormat::RGB, TextureComponentSize::BITS_16, TextureComponentType::FLOAT, _state.windowWidth, _state.windowHeight, false}, nullptr);
    _state.lightingColorBuffer.setMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);
    _state.lightingColorBuffer.setCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);

    // Create the buffer we will use to add bloom as a post-processing effect
    _state.lightingHighBrightnessBuffer = Texture(TextureConfig{TextureType::TEXTURE_2D, TextureComponentFormat::RGB, TextureComponentSize::BITS_16, TextureComponentType::FLOAT, _state.windowWidth, _state.windowHeight, false}, nullptr);
    _state.lightingHighBrightnessBuffer.setMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);
    _state.lightingHighBrightnessBuffer.setCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);

    // Create the depth buffer
    _state.lightingDepthBuffer = Texture(TextureConfig{TextureType::TEXTURE_2D, TextureComponentFormat::DEPTH, TextureComponentSize::BITS_DEFAULT, TextureComponentType::FLOAT, _state.windowWidth, _state.windowHeight, false}, nullptr);
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
    for (; _state.numDownsampleIterations < 6; ++_state.numDownsampleIterations) {
        currWidth /= 2;
        currHeight /= 2;
        if (currWidth < 8 || currHeight < 8) break;
        PostFXBuffer buffer;
        auto color = Texture(TextureConfig{ TextureType::TEXTURE_2D, TextureComponentFormat::RGB, TextureComponentSize::BITS_16, TextureComponentType::FLOAT, currWidth, currHeight, false }, nullptr);
        color.setMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);
        color.setCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE); // TODO: Does this make sense for bloom textures?
        buffer.fbo = FrameBuffer({ color });
        if (!buffer.fbo.valid()) {
            _isValid = false;
            std::cerr << "Unable to initialize bloom buffer" << std::endl;
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
        auto color = Texture(TextureConfig{ TextureType::TEXTURE_2D, TextureComponentFormat::RGB, TextureComponentSize::BITS_16, TextureComponentType::FLOAT, width, height, false }, nullptr);
        color.setMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);
        color.setCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE); // TODO: Does this make sense for bloom textures?
        buffer.fbo = FrameBuffer({ color });
        if (!buffer.fbo.valid()) {
            _isValid = false;
            std::cerr << "Unable to initialize bloom buffer" << std::endl;
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

void Renderer::setRenderMode(RenderMode mode) {
    _state.mode = mode;
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

        for (auto& csm : _state.csms) {
            csm.fbo.clear(color);
        }

        for (auto& gaussian : _state.gaussianBuffers) {
            gaussian.fbo.clear(color);
        }

        for (auto& postFx : _state.postFxBuffers) {
            postFx.fbo.clear(color);
        }
    }

    // Clear all entities from the previous frame
    for (auto & e : _state.entities) {
        e.second.clear();
    }

    // Clear all instanced entities
    _state.instancedMeshes.clear();

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

void Renderer::addDrawable(RenderEntity * e) {
    _addDrawable(e, glm::mat4(1.0f));
}

static void addEntityMeshData(RenderEntity * e, std::unordered_map<__RenderEntityObserver, std::unordered_map<__MeshObserver, __MeshContainer>> & map) {
    __RenderEntityObserver c(e);
    if (map.find(c) == map.end()) {
        map.insert(std::make_pair(c, std::unordered_map<__MeshObserver, __MeshContainer>{}));
    }
    std::unordered_map<__MeshObserver, __MeshContainer> & existing = map.find(c)->second;
    
    for (std::shared_ptr<Mesh> m : e->meshes) {
        __MeshObserver o(m.get());
        if (existing.find(o) == existing.end()) {
            existing.insert(std::make_pair(o, __MeshContainer(m.get())));
        }
        __MeshContainer & container = existing.find(o)->second;
        container.modelMatrices.push_back(e->model);
        container.diffuseColors.push_back(m->getMaterial().diffuseColor);
        container.baseReflectivity.push_back(m->getMaterial().baseReflectivity);
        container.roughness.push_back(m->getMaterial().roughness);
        container.metallic.push_back(m->getMaterial().metallic);
        ++container.size;
    }
}

void Renderer::_addDrawable(RenderEntity * e, const glm::mat4 & accum) {
    auto it = _state.entities.find(e->getLightProperties());
    if (it == _state.entities.end()) {
        // Not necessarily an error since if an entity is set to
        // invisible, we won't bother adding them
        //std::cerr << "[error] Unable to add entity" << std::endl;
        return;
    }
    e->model = glm::mat4(1.0f);
    matRotate(e->model, e->rotation);
    matScale(e->model, e->scale);
    matTranslate(e->model, e->position);
    e->model = accum * e->model;
    it->second.push_back(e);
    if (e->getLightProperties() & DYNAMIC) {
        _state.lightInteractingEntities.push_back(e);
    }

    // We want to keep track of entities and whether or not they have moved for determining
    // when shadows should be recomputed
    if (_entitiesSeenBefore.find(e) == _entitiesSeenBefore.end()) {
        _entitiesSeenBefore.insert(std::make_pair(e, EntityStateInfo{e->position, e->scale, e->rotation.asVec3(), true}));
    }
    else {
        EntityStateInfo & info = _entitiesSeenBefore.find(e)->second;
        const double distance = glm::distance(e->position, info.lastPos);
        const double scale = glm::distance(e->scale, info.lastScale);
        const double rotation = glm::distance(e->rotation.asVec3(), info.lastRotation);
        info.dirty = false;
        if (distance > 0.25) {
            info.lastPos = e->position;
            info.dirty = true;
        }
        if (scale > 0.25) {
            info.lastScale = e->scale;
            info.dirty = true;
        }
        if (rotation > 0.25) {
            info.lastRotation = e->rotation.asVec3();
            info.dirty = true;
        }
    }

    //addEntityMeshData(e, _state.instancedMeshes);

    for (RenderEntity & node : e->nodes) {
        _addDrawable(&node, e->model);
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

void Renderer::_initInstancedData(__MeshContainer & c, std::vector<GLuint> & buffers) {
    Pipeline * pbr = _state.geometry.get();

    auto & modelMats = c.modelMatrices;
    auto & diffuseColors = c.diffuseColors;
    auto & baseReflectivity = c.baseReflectivity;
    auto & roughness = c.roughness;
    auto & metallic = c.metallic;

    // All shaders should use the same location for model, so this should work
    int pos = pbr->getAttribLocation("model");
    const int pos1 = pos + 0;
    const int pos2 = pos + 1;
    const int pos3 = pos + 2;
    const int pos4 = pos + 3;

    c.m->bind();

    GLuint buffer;
    glGenBuffers(1, &buffer);
    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    glBufferData(GL_ARRAY_BUFFER, modelMats.size() * sizeof(glm::mat4), &modelMats[0], GL_STATIC_DRAW);

    glEnableVertexAttribArray(pos1);
    glVertexAttribPointer(pos1, 4, GL_FLOAT, GL_FALSE, 64, (void *)0);
    glEnableVertexAttribArray(pos2);
    glVertexAttribPointer(pos2, 4, GL_FLOAT, GL_FALSE, 64, (void *)16);
    glEnableVertexAttribArray(pos3);
    glVertexAttribPointer(pos3, 4, GL_FLOAT, GL_FALSE, 64, (void *)32);
    glEnableVertexAttribArray(pos4);
    glVertexAttribPointer(pos4, 4, GL_FLOAT, GL_FALSE, 64, (void *)48);
    glVertexAttribDivisor(pos1, 1);
    glVertexAttribDivisor(pos2, 1);
    glVertexAttribDivisor(pos3, 1);
    glVertexAttribDivisor(pos4, 1);

    buffers.push_back(buffer);

    buffer = 0;
    glGenBuffers(1, &buffer);
    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    glBufferData(GL_ARRAY_BUFFER, diffuseColors.size() * sizeof(glm::vec3), &diffuseColors[0], GL_STATIC_DRAW);
    pos = pbr->getAttribLocation("diffuseColor");
    glEnableVertexAttribArray(pos);
    glVertexAttribPointer(pos, 3, GL_FLOAT, GL_FALSE, 0, (void *)0);
    glVertexAttribDivisor(pos, 1);
    buffers.push_back(buffer);

    buffer = 0;
    glGenBuffers(1, &buffer);
    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    glBufferData(GL_ARRAY_BUFFER, baseReflectivity.size() * sizeof(glm::vec3), &baseReflectivity[0], GL_STATIC_DRAW);
    pos = pbr->getAttribLocation("baseReflectivity");
    glEnableVertexAttribArray(pos);
    glVertexAttribPointer(pos, 3, GL_FLOAT, GL_FALSE, 0, (void *)0);
    glVertexAttribDivisor(pos, 1);
    buffers.push_back(buffer);

    buffer = 0;
    glGenBuffers(1, &buffer);
    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    glBufferData(GL_ARRAY_BUFFER, metallic.size() * sizeof(float), &metallic[0], GL_STATIC_DRAW);
    // All shaders should use the same location for shininess, so this should work
    pos = pbr->getAttribLocation("metallic");
    glEnableVertexAttribArray(pos);
    glVertexAttribPointer(pos, 1, GL_FLOAT, GL_FALSE, 0, (void *)0);
    glVertexAttribDivisor(pos, 1);
    buffers.push_back(buffer);

    buffer = 0;
    glGenBuffers(1, &buffer);
    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    glBufferData(GL_ARRAY_BUFFER, roughness.size() * sizeof(float), &roughness[0], GL_STATIC_DRAW);
    // All shaders should use the same location for shininess, so this should work
    pos = pbr->getAttribLocation("roughness");
    glEnableVertexAttribArray(pos);
    glVertexAttribPointer(pos, 1, GL_FLOAT, GL_FALSE, 0, (void *)0);
    glVertexAttribDivisor(pos, 1);
    buffers.push_back(buffer);

    c.m->unbind();
}

void Renderer::_clearInstancedData(std::vector<GLuint> & buffers) {
    glDeleteBuffers(buffers.size(), &buffers[0]);
    buffers.clear();
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
    for (auto & entityList : _state.entities) {
        for (RenderEntity * e : entityList.second) {
            const double distance = glm::distance(e->position, c.getPosition());
            if (distance < _state.zfar) {
                addEntityMeshData(e, _state.instancedMeshes);
            }
        }
    }
}

void Renderer::_render(const Camera & c, const RenderEntity * e, const Mesh * m, const size_t numInstances, bool removeViewTranslation) {
    const glm::mat4 & projection = _state.perspective;
    //const glm::mat4 & view = c.getViewTransform();
    glm::mat4 view;
    if (removeViewTranslation) {
        // Remove the translation component of the view matrix
        view = glm::mat4(glm::mat3(c.getViewTransform()));
    }
    else {
        view = c.getViewTransform();
    }

    // Unbind current shader if one is bound
    _unbindShader();

    // Set up the shader we will use for this batch of entities
    Pipeline * s;
    uint32_t lightProperties = e->getLightProperties();
    uint32_t renderProperties = m->getRenderProperties();
    if (lightProperties == FLAT) {
        s = _state.forward.get();
    }
    else {
        s = _state.geometry.get();
    }

    //s->print();
    _bindShader(s);

    s->setMat4("projection", &projection[0][0]);
    s->setMat4("view", &view[0][0]);

    if (renderProperties & TEXTURED) {
        //_bindTexture(s, "diffuseTexture", m->getMaterial().texture);
        s->bindTexture("diffuseTexture", _lookupTexture(m->getMaterial().texture));
        s->setBool("textured", true);
    }
    else {
        s->setBool("textured", false);
    }

    // Determine which uniforms we should set
    if (lightProperties & FLAT) {
        s->setVec3("diffuseColor", &m->getMaterial().diffuseColor[0]);
    } else if (lightProperties & DYNAMIC) {
        if (renderProperties & NORMAL_MAPPED) {
            //_bindTexture(s, "normalMap", m->getMaterial().normalMap);
            s->bindTexture("normalMap", _lookupTexture(m->getMaterial().normalMap));
            s->setBool("normalMapped", true);
        }
        else {
            s->setBool("normalMapped", false);
        }

        if (renderProperties & HEIGHT_MAPPED) {
            //_bindTexture(s, "depthMap", m->getMaterial().depthMap);
            s->bindTexture("depthMap", _lookupTexture(m->getMaterial().depthMap));
            s->setFloat("heightScale", m->getMaterial().heightScale);
            s->setBool("depthMapped", true);
        }
        else {
            s->setBool("depthMapped", false);
        }

        if (renderProperties & ROUGHNESS_MAPPED) {
            //_bindTexture(s, "roughnessMap", m->getMaterial().roughnessMap);
            s->bindTexture("roughnessMap", _lookupTexture(m->getMaterial().roughnessMap));
            s->setBool("roughnessMapped", true);
        }
        else {
            s->setBool("roughnessMapped", false);
        }

        if (renderProperties & AMBIENT_MAPPED) {
            //_bindTexture(s, "ambientOcclusionMap", m->getMaterial().ambientMap);
            s->bindTexture("ambientOcclusionMap", _lookupTexture(m->getMaterial().ambientMap));
            s->setBool("ambientMapped", true);
        }
        else {
            s->setBool("ambientMapped", false);
        }

        if (renderProperties & SHININESS_MAPPED) {
            //_bindTexture(s, "metalnessMap", m->getMaterial().metalnessMap);
            s->bindTexture("metalnessMap", _lookupTexture(m->getMaterial().metalnessMap));
            s->setBool("metalnessMapped", true);
        }
        else {
            s->setBool("metalnessMapped", false);
        }

        s->setVec3("viewPosition", &c.getPosition()[0]);
    }

    // Perform instanced rendering
    setCullState(m->cullingMode);

    m->bind();
    m->render(numInstances);
    m->unbind();

    _unbindShader();
}

void Renderer::_renderCSMDepth(const Camera & c, const std::unordered_map<__RenderEntityObserver, std::unordered_map<__MeshObserver, __MeshContainer>> & entities) {
    _bindShader(_state.csmDepth.get());
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    //glBlendFunc(GL_ONE, GL_ONE);
    // glDisable(GL_CULL_FACE);
    for (CascadedShadowMap & csm : _state.csms) {
        csm.fbo.bind();
        _state.csmDepth->setMat4("projection", &csm.depthProjection[0][0]);
        _state.csmDepth->setMat4("view", &csm.view[0][0]);
        const Texture * depth = csm.fbo.getDepthStencilAttachment();
        if (!depth) {
            throw std::runtime_error("Critical error: depth attachment not present");
        }
        glViewport(0, 0, depth->width(), depth->height());
        // Render each entity into the depth map
        for (auto & entityObservers : entities) {
            for (auto & meshObservers : entityObservers.second) {
                const RenderEntity * e = entityObservers.first.e;
                const Mesh * m = meshObservers.first.m;
                const size_t numInstances = meshObservers.second.size;
                setCullState(m->cullingMode);
                m->bind();
                m->render(numInstances);
                m->unbind();
            }
        }
        csm.fbo.unbind();
    }
    _unbindShader();
}

void Renderer::end(const Camera & c) {
    //if (_state.worldLightingEnabled) {
    //    _recalculateCascadeData(c);
    //}
    // TODO: Recalculate every scene?
    _recalculateCascadeData(c);

    // Pull the view transform/projection matrices
    // const glm::mat4 * projection = &_state.perspective;
    // const glm::mat4 * view = &c.getViewTransform();

    // const glm::mat4 cameraToWorld = glm::inverse(*view);
    // glm::mat4 lightToWorld = glm::mat4(1.0f);
    // lightToWorld[3] = glm::vec4(glm::vec3(0.0f, 10.0f, 0.0f), 1.0f);
    // const glm::mat4 lightViewMat = glm::inverse(lightToWorld);
    // Camera lightCam;
    // lightCam.setDirection(_state.worldLight.getDirection());
    // std::cout << lightCam.getViewTransform() << std::endl;

    const int maxInstances = 250;
    const int maxShadowCastingLights = 8;
    const int maxTotalLights = 256;
    const int maxShadowUpdatesPerFrame = maxShadowCastingLights;
    // Need to delete these at the end of the frame
    std::vector<GLuint> buffers;

    //_unbindAllTextures();

    // We need to figure out what we want to attempt to render
    _buildEntityList(c);

    std::unordered_map<Light *, std::unordered_map<__RenderEntityObserver, std::unordered_map<__MeshObserver, __MeshContainer>>> perLightInstancedMeshes;
    std::unordered_map<Light *, bool> perLightIsDirty;
    std::vector<std::pair<Light *, double>> perLightDistToViewer;
    // This one is just for shadow-casting lights
    std::vector<std::pair<Light *, double>> perLightShadowCastingDistToViewer;
    // Init per light instance data
    for (Light * light : _state.lights) {
        const double distance = glm::distance(c.getPosition(), light->position);
        perLightDistToViewer.push_back(std::make_pair(light, distance));
        //if (distance > 2 * light->getRadius()) continue;
        perLightInstancedMeshes.insert(std::make_pair(light, std::unordered_map<__RenderEntityObserver, std::unordered_map<__MeshObserver, __MeshContainer>>()));
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

    for (RenderEntity * e : _state.lightInteractingEntities) {
        const bool entityIsDirty = _entitiesSeenBefore.find(e)->second.dirty;
        for (auto&[light, _] : perLightShadowCastingDistToViewer) {
            const double distance = glm::distance(e->position, light->position);
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
        for (auto & entityObservers : instancedMeshes) {
            for (auto & meshObservers : entityObservers.second) {
                _initInstancedData(meshObservers.second, buffers);
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

        /*
        for (auto & p : _state.entities) {
            uint32_t properties = p.first;
            if ( !(properties & DYNAMIC) ) continue;
            for (auto & e : p.second) {
                _state.shadows->setMat4("model", &e->model[0][0]);
                e->render();
            }
        }
        */

        for (auto & entityObservers : instancedMeshes) {
            //uint32_t properties = entityObservers.first.e->getLightProperties();
            //if ( !(properties & DYNAMIC) ) continue;
            for (auto & meshObservers : entityObservers.second) {
                // Set up temporary instancing buffers
                //_initInstancedData(meshObservers.second, buffers);
                Mesh * m = meshObservers.first.m;
                setCullState(m->cullingMode);
                m->bind();
                m->render(meshObservers.second.size);
                m->unbind();
                //_clearInstancedData(buffers);
                /**
                const size_t size = modelMats.size();
                for (int i = 0; i < size; i += maxInstances) {
                    const size_t instances = std::min<size_t>(maxInstances, size - i);
                    _state.shadows->setMat4("modelMats", &modelMats[i][0][0], instances);
                    e.second.e->render(instances);
                }
                */
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
    for (auto & entityObservers : _state.instancedMeshes) {
        for (auto & meshObservers : entityObservers.second) {
            _initInstancedData(meshObservers.second, buffers);
        }
    }

    // Perform world light depth pass if enabled
    if (_state.worldLightingEnabled) {
        _renderCSMDepth(c, _state.instancedMeshes);
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
    for (auto & entityObservers : _state.instancedMeshes) {
        for (auto & meshObservers : entityObservers.second) {
            //_initInstancedData(meshObservers.second, buffers);
            RenderEntity * e = entityObservers.first.e;
            Mesh * m = meshObservers.first.m;
            const size_t numInstances = meshObservers.second.size;

            // We are only going to render dynamic-lit entities this pass
            if (e->getLightProperties() & FLAT) continue;
            _render(c, e, m, numInstances);
        }
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
    _state.screenQuad->bind();
    _state.screenQuad->render(1);
    _state.screenQuad->unbind();
    _state.lightingFbo.unbind();
    _unbindShader();

    // Forward pass for all objects that don't interact with light (may also be used for transparency later as well)
    _state.lightingFbo.copyFrom(_state.buffer.fbo, BufferBounds{0, 0, _state.windowWidth, _state.windowHeight}, BufferBounds{0, 0, _state.windowWidth, _state.windowHeight}, BufferBit::DEPTH_BIT, BufferFilter::NEAREST);
    // Blit to default framebuffer - not that the framebuffer you are writing to has to match the internal format
    // of the framebuffer you are reading to!
    glEnable(GL_DEPTH_TEST);
    _state.lightingFbo.bind();
    for (auto & entityObservers : _state.instancedMeshes) {
        for (auto & meshObservers : entityObservers.second) {
            //_initInstancedData(meshObservers.second, buffers);
            RenderEntity * e = entityObservers.first.e;
            Mesh * m = meshObservers.first.m;
            const size_t numInstances = meshObservers.second.size;

            // We are only going to render flat entities during this pass
            if (e->getLightProperties() & DYNAMIC) continue;
            _render(c, e, m, numInstances);
        }
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
    _state.screenQuad->bind();
    _state.screenQuad->render(1);
    _state.screenQuad->unbind();
}

static Texture _loadTexture(const std::string & file) {
    Texture texture;
    int width, height, numChannels;
    // @see http://www.redbancosdealimentos.org/homes-flooring-design-sources
    uint8_t * data = stbi_load(file.c_str(), &width, &height, &numChannels, 0);
    if (data) {
        TextureConfig config;
        config.type = TextureType::TEXTURE_2D;
        config.storage = TextureComponentSize::BITS_DEFAULT;
        config.generateMipMaps = true;
        config.dataType = TextureComponentType::UINT;
        config.width = (uint32_t)width;
        config.height = (uint32_t)height;
        // This loads the textures with sRGB in mind so that they get converted back
        // to linear color space. Warning: if the texture was not actually specified as an
        // sRGB texture (common for normal/specular maps), this will cause problems.
        switch (numChannels) {
            case 1:
                config.format = TextureComponentFormat::RED;
                break;
            case 3:
                config.format = TextureComponentFormat::SRGB;
                break;
            case 4:
                config.format = TextureComponentFormat::SRGB_ALPHA;
                break;
            default:
                std::cerr << "[error] Unknown texture loading error - format may be invalid" << std::endl;
                stbi_image_free(data);
                return Texture();
        }

        texture = Texture(config, data);
        texture.setCoordinateWrapping(TextureCoordinateWrapping::REPEAT);
        texture.setMinMagFilter(TextureMinificationFilter::LINEAR_MIPMAP_LINEAR, TextureMagnificationFilter::LINEAR);
    } else {
        std::cerr << "[error] Could not load texture: " << file << std::endl;
        return Texture();
    }
    
    stbi_image_free(data);
    return texture;
}

TextureHandle Renderer::loadTexture(const std::string &file) {
    auto it = _textures.find(file);
    if (it != _textures.end()) return it->second.handle;

    TextureCache tex;
    tex.file = file;
    tex.handle = this->_nextTextureHandle++;
    tex.texture = _loadTexture(file);
    if (!tex.texture.valid()) return -1;

    _textures.insert(std::make_pair(file, tex));
    _textureHandles.insert(std::make_pair(tex.handle, tex));
    return tex.handle;
}

Model Renderer::loadModel(const std::string & file) {
    auto it = this->_models.find(file);
    if (it != this->_models.end()) {
        return it->second;
    }

    std::cout << "Loading " << file << std::endl;
    Model m(*this, file);
    this->_models.insert(std::make_pair(file, m));
    return std::move(m);
}

ShadowMapHandle Renderer::createShadowMap3D(uint32_t resolutionX, uint32_t resolutionY) {
    ShadowMap3D smap;
    smap.shadowCubeMap = Texture(TextureConfig{TextureType::TEXTURE_3D, TextureComponentFormat::DEPTH, TextureComponentSize::BITS_DEFAULT, TextureComponentType::FLOAT, resolutionX, resolutionY, false}, nullptr);
    smap.shadowCubeMap.setMinMagFilter(TextureMinificationFilter::NEAREST, TextureMagnificationFilter::NEAREST);
    smap.shadowCubeMap.setCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);

    smap.frameBuffer = FrameBuffer({smap.shadowCubeMap});
    if (!smap.frameBuffer.valid()) {
        _isValid = false;
        return -1;
    }
    TextureHandle handle = this->_nextTextureHandle++;
    this->_shadowMap3DHandles.insert(std::make_pair(handle, smap));
    return handle;
}

void Renderer::invalidateAllTextures() {
    for (auto & texture : _textures) {
        //glDeleteTextures(1, &texture.second.texture);
        texture.second.texture = Texture();
        // Make sure we mark it as unloaded just in case someone tries
        // to use it in the future
        texture.second.loaded = false;
    }
}

Texture Renderer::_lookupTexture(TextureHandle handle) const {
    if (handle == -1) return Texture();

    auto it = _textureHandles.find(handle);
    // TODO: Make sure that 0 actually signifies an invalid texture in OpenGL
    if (it == _textureHandles.end()) {
        if (_shadowMap3DHandles.find(handle) == _shadowMap3DHandles.end()) return Texture();
        return _shadowMap3DHandles.find(handle)->second.shadowCubeMap;
    }

    // If not in memory then bring it in
    if (!it->second.loaded) {
        TextureCache tex = it->second;
        tex.texture = _loadTexture(tex.file);
        tex.loaded = tex.texture.valid();
        _textures.erase(tex.file);
        _textures.insert(std::make_pair(tex.file, tex));
        _textureHandles.erase(handle);
        _textureHandles.insert(std::make_pair(handle, tex));
        return tex.texture;
    }
    return it->second.texture;
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
            s->bindTexture("shadowCubeMaps[" + std::to_string(shadowLightIndex) + "]", _lookupTexture(_getShadowMapHandleForLight(light)));
            ++shadowLightIndex;
        }
        ++lightIndex;
    }

    if (shadowLightIndex == 0) {
       // If we don't do this the fragment shader crashes
       s->setFloat("lightFarPlanes[0]", 0.0f);
       //_bindShadowMapTexture(s, "shadowCubeMaps[0]", _state.dummyCubeMap);
       s->bindTexture("shadowCubeMaps[0]", _lookupTexture(_state.dummyCubeMap));
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
    Camera lightCam(false);
    lightCam.setAngle(_state.worldLight.getRotation());
    glm::mat4 lightWorld = lightCam.getWorldTransform();
    glm::mat4 lightView = lightCam.getViewTransform();
    glm::vec3 direction = lightCam.getDirection(); //glm::vec3(-lightWorld[2].x, -lightWorld[2].y, -lightWorld[2].z);
    std::cout << "Light direction: " << direction << std::endl;
    s->setBool("infiniteLightingEnabled", _state.worldLightingEnabled);
    s->setVec3("infiniteLightDirection", &direction[0]);
    lightColor = _state.worldLight.getColor() * _state.worldLight.getIntensity();
    s->setVec3("infiniteLightColor", &lightColor[0]);

    for (int i = 0; i < 4; ++i) {
        s->bindTexture("infiniteLightShadowMaps[" + std::to_string(i) + "]", *_state.csms[i].fbo.getDepthStencilAttachment());
        s->setMat4("cascadeProjViews[" + std::to_string(i) + "]", &_state.csms[i].projectionView[0][0]);
    }

    for (int i = 0; i < 2; ++i) {
        s->setVec4("shadowOffset[" + std::to_string(i) + "]", &_state.cascadeShadowOffsets[i][0]);
    }

    for (int i = 0; i < 3; ++i) {
        s->setVec3("cascadeScale[" + std::to_string(i) + "]", &_state.csms[i + 1].cascadeScale[0]);
        s->setVec3("cascadeOffset[" + std::to_string(i) + "]", &_state.csms[i + 1].cascadeOffset[0]);
        s->setVec4("cascadePlanes[" + std::to_string(i) + "]", &_state.csms[i + 1].cascadePlane[0]);
    }

    // s->setMat4("cascade0ProjView", &_state.csms[0].projectionView[0][0]);
}

ShadowMapHandle Renderer::_getShadowMapHandleForLight(Light * light) {
    assert(_shadowMap3DHandles.size() > 0);

    auto it = _lightsToShadowMap.find(light);
    // If not found, look for an existing shadow map
    if (it == _lightsToShadowMap.end()) {
        // Mark the light as dirty since its map will need to be updated
        _lightsSeenBefore.find(light)->second.dirty = true;

        ShadowMapHandle handle = -1;
        for (const auto & entry : _shadowMap3DHandles) {
            if (_usedShadowMaps.find(entry.first) == _usedShadowMaps.end()) {
                handle = entry.first;
                break;
            }
        }

        if (handle == -1) {
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

void Renderer::_setLightShadowMapHandle(Light * light, ShadowMapHandle handle) {
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
}
}