
#include <StratusRendererBackend.h>
#include <iostream>
#include <StratusLight.h>
#include "StratusPipeline.h"
#include "StratusRendererBackend.h"
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

RendererBackend::RendererBackend(const uint32_t width, const uint32_t height, const std::string& appName) {
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        STRATUS_ERROR << "Unable to initialize sdl2" << std::endl;
        return;
    }

    _window = SDL_CreateWindow(appName.c_str(),
            100, 100, // location x/y on screen
            width, height, // width/height of window
            SDL_WINDOW_SHOWN | SDL_WINDOW_OPENGL );
    if (_window == nullptr) {
        STRATUS_ERROR << "Failed to create sdl window" << std::endl;
        SDL_Quit();
        return;
    }

    //const int32_t minGLVersion = 2;

    // Set the profile to core as opposed to immediate mode
    SDL_GL_SetAttribute(SDL_GLattr::SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    // Set max/min version to be 3.2
    //SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, maxGLVersion);
    //SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, minGLVersion);
    // Enable double buffering
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

    // Create the gl context
    _context = SDL_GL_CreateContext(_window);
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
    _ValidateAllShaders();

    _state.dummyCubeMap = CreateShadowMap3D(_state.shadowCubeMapX, _state.shadowCubeMapY);

    // Create a pool of shadow maps for point lights to use
    for (int i = 0; i < _state.numShadowMaps; ++i) {
        CreateShadowMap3D(_state.shadowCubeMapX, _state.shadowCubeMapY);
    }
}

void RendererBackend::_ValidateAllShaders() {
    _isValid = true;
    for (Pipeline * p : _state.shaders) {
        _isValid = _isValid && p->isValid();
    }
}

RendererBackend::~RendererBackend() {
    if (_context) {
        SDL_GL_DeleteContext(_context);
        _context = nullptr;
    }

    if (_window) {
        SDL_DestroyWindow(_window);
        _window = nullptr;
        SDL_Quit();
    }

    for (Pipeline * shader : _shaders) delete shader;
    _shaders.clear();

    // Delete the main frame buffer
    _ClearGBuffer();
}

void RendererBackend::RecompileShaders() {
    for (Pipeline* p : _state.shaders) {
        p->recompile();
    }
    _ValidateAllShaders();
}

const GFXConfig & RendererBackend::Config() const {
    return _config;
}

bool RendererBackend::Valid() const {
    return _isValid;
}

const Pipeline *RendererBackend::GetCurrentShader() const {
    return nullptr;
}

void RendererBackend::_RecalculateCascadeData() {
    const uint32_t cascadeResolutionXY = _frame->csc.cascadeResolutionXY;
    const uint32_t numCascades = _frame->csc.cascades.size();
    if (_frame->csc.regenerateFbo || !_frame->csc.fbo.valid()) {
        // Create the depth buffer
        // @see https://stackoverflow.com/questions/22419682/glsl-sampler2dshadow-and-shadow2d-clarificationssss
        Texture tex(TextureConfig{ TextureType::TEXTURE_2D_ARRAY, TextureComponentFormat::DEPTH, TextureComponentSize::BITS_DEFAULT, TextureComponentType::FLOAT, cascadeResolutionXY, cascadeResolutionXY, numCascades, false }, nullptr);
        tex.setMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);
        tex.setCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);
        // We need to set this when using sampler2DShadow in the GLSL shader
        tex.setTextureCompare(TextureCompareMode::COMPARE_REF_TO_TEXTURE, TextureCompareFunc::LEQUAL);

        // Create the frame buffer
        _frame->csc.fbo = FrameBuffer({ tex });
    }
}

void RendererBackend::_ClearGBuffer() {
    _state.buffer = GBuffer();
    _state.gaussianBuffers.clear();
    _state.postFxBuffers.clear();
}

void RendererBackend::_UpdateWindowDimensions() {
    if ( !_frame->viewportDirty ) return;
    glViewport(0, 0, _frame->viewportWidth, _frame->viewportHeight);

    // Regenerate the main frame buffer
    _ClearGBuffer();

    GBuffer & buffer = _state.buffer;
    // glGenFramebuffers(1, &buffer.fbo);
    // glBindFramebuffer(GL_FRAMEBUFFER, buffer.fbo);

    // Position buffer
    buffer.position = Texture(TextureConfig{TextureType::TEXTURE_2D, TextureComponentFormat::RGB, TextureComponentSize::BITS_32, TextureComponentType::FLOAT, _frame->viewportWidth, _frame->viewportHeight, 0, false}, nullptr);
    buffer.position.setMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);

    // Normal buffer
    buffer.normals = Texture(TextureConfig{TextureType::TEXTURE_2D, TextureComponentFormat::RGB, TextureComponentSize::BITS_32, TextureComponentType::FLOAT, _frame->viewportWidth, _frame->viewportHeight, 0, false}, nullptr);
    buffer.normals.setMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);

    // Create the color buffer - notice that is uses higher
    // than normal precision. This allows us to write color values
    // greater than 1.0 to support things like HDR.
    buffer.albedo = Texture(TextureConfig{TextureType::TEXTURE_2D, TextureComponentFormat::RGB, TextureComponentSize::BITS_16, TextureComponentType::FLOAT, _frame->viewportWidth, _frame->viewportHeight, 0, false}, nullptr);
    buffer.albedo.setMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);

    // Base reflectivity buffer
    buffer.baseReflectivity = Texture(TextureConfig{TextureType::TEXTURE_2D, TextureComponentFormat::RGB, TextureComponentSize::BITS_16, TextureComponentType::FLOAT, _frame->viewportWidth, _frame->viewportHeight, 0, false}, nullptr);
    buffer.baseReflectivity.setMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);

    // Roughness-Metallic-Ambient buffer
    buffer.roughnessMetallicAmbient = Texture(TextureConfig{TextureType::TEXTURE_2D, TextureComponentFormat::RGB, TextureComponentSize::BITS_16, TextureComponentType::FLOAT, _frame->viewportWidth, _frame->viewportHeight, 0, false}, nullptr);
    buffer.roughnessMetallicAmbient.setMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);

    // Create the depth buffer
    buffer.depth = Texture(TextureConfig{TextureType::TEXTURE_2D, TextureComponentFormat::DEPTH, TextureComponentSize::BITS_DEFAULT, TextureComponentType::FLOAT, _frame->viewportWidth, _frame->viewportHeight, 0, false}, nullptr);
    buffer.depth.setMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);

    // Create the frame buffer with all its texture attachments
    buffer.fbo = FrameBuffer({buffer.position, buffer.normals, buffer.albedo, buffer.baseReflectivity, buffer.roughnessMetallicAmbient, buffer.depth});
    if (!buffer.fbo.valid()) {
        _isValid = false;
        return;
    }

    // Code to create the lighting fbo
    _state.lightingColorBuffer = Texture(TextureConfig{TextureType::TEXTURE_2D, TextureComponentFormat::RGB, TextureComponentSize::BITS_16, TextureComponentType::FLOAT, _frame->viewportWidth, _frame->viewportHeight, 0, false}, nullptr);
    _state.lightingColorBuffer.setMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);
    _state.lightingColorBuffer.setCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);

    // Create the buffer we will use to add bloom as a post-processing effect
    _state.lightingHighBrightnessBuffer = Texture(TextureConfig{TextureType::TEXTURE_2D, TextureComponentFormat::RGB, TextureComponentSize::BITS_16, TextureComponentType::FLOAT, _frame->viewportWidth, _frame->viewportHeight, 0, false}, nullptr);
    _state.lightingHighBrightnessBuffer.setMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);
    _state.lightingHighBrightnessBuffer.setCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);

    // Create the depth buffer
    _state.lightingDepthBuffer = Texture(TextureConfig{TextureType::TEXTURE_2D, TextureComponentFormat::DEPTH, TextureComponentSize::BITS_DEFAULT, TextureComponentType::FLOAT, _frame->viewportWidth, _frame->viewportHeight, 0, false}, nullptr);
    _state.lightingDepthBuffer.setMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);

    // Attach the textures to the FBO
    _state.lightingFbo = FrameBuffer({_state.lightingColorBuffer, _state.lightingHighBrightnessBuffer, _state.lightingDepthBuffer});
    if (!_state.lightingFbo.valid()) {
        _isValid = false;
        return;
    }

    _InitializePostFxBuffers();
}

void RendererBackend::_InitializePostFxBuffers() {
    uint32_t currWidth = _frame->viewportWidth;
    uint32_t currHeight = _frame->viewportHeight;
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
    sizes.push_back(std::make_pair(_frame->viewportWidth, _frame->viewportHeight));
    
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

void RendererBackend::_ClearFramebufferData(const bool clearScreen) {
    // Always clear the main screen buffer, but only
    // conditionally clean the custom frame buffer
    glClearColor(_frame->clearColor.r, _frame->clearColor.g, _frame->clearColor.b, _frame->clearColor.a);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (clearScreen) {
        const glm::vec4& color = _frame->clearColor;
        _state.buffer.fbo.clear(color);
        _state.lightingFbo.clear(color);

        // Depending on when this happens we may not have generated cascadeFbo yet
        if (_frame->csc.fbo.valid()) {
            _frame->csc.fbo.clear(glm::vec4(0.0f));
        }

        for (auto& gaussian : _state.gaussianBuffers) {
            gaussian.fbo.clear(glm::vec4(0.0f));
        }

        for (auto& postFx : _state.postFxBuffers) {
            postFx.fbo.clear(glm::vec4(0.0f));
        }
    }
}

void RendererBackend::_InitAllInstancedData() {
#define INIT_INST_DATA(map)                                         \
    for (auto& entry : map) {                                       \
        RenderNodeView node = entry.first;                          \
        std::vector<RendererEntityData>& dataVec = entry.second;    \
        for (auto& data : dataVec) {                                \
            _InitInstancedData(data);                               \
        }                                                           \
    }

    // Dynamic entities
    INIT_INST_DATA(_frame->instancedPbrMeshes)

    // Flat entities
    INIT_INST_DATA(_frame->instancedFlatMeshes)

    // Shadow-casting lights
    for (auto& entry : _frame->lights) {
        auto light = entry.first;
        auto& lightData = entry.second;
        if (light->castsShadows() && lightData.dirty) {
            INIT_INST_DATA(lightData.visible)
        }
    }

    // Cascades
    if (_frame->csc.worldLightingEnabled) {
        INIT_INST_DATA(_frame->csc.visible)
    }

#undef INIT_INST_DATA
}

void RendererBackend::Begin(const std::shared_ptr<RendererFrame>& frame, bool clearScreen) {
    _frame = frame;

    // Make sure we set our context as the active one
    SDL_GL_MakeCurrent(_window, _context);

    // Clear out instanced data from previous frame
    _ClearInstancedData();

    // Checks to see if any framebuffers need to be generated or re-generated
    _RecalculateCascadeData();

    // Update all dimension, texture and framebuffer data if the viewport changed
    _UpdateWindowDimensions();

    // Includes screen data
    _ClearFramebufferData(clearScreen);

    // Generate the GPU data for all instanced entities
    _InitAllInstancedData();

    // Collect window input events
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
        _state.events.push_back(e);
    }

    // Update mouse
    _state.mouse.mask = SDL_GetMouseState(&_state.mouse.x, &_state.mouse.y);

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

/**
 * During the lighting phase, we need each of the 6 faces of the shadow map to have its own view transform matrix.
 * This enables us to convert vertices to be in various different light coordinate spaces.
 */
static std::vector<glm::mat4> GenerateLightViewTransforms(const glm::mat4 & projection, const glm::vec3 & lightPos) {
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

void RendererBackend::_InitInstancedData(RendererEntityData & c) {
    Pipeline * pbr = _state.geometry.get();

    auto & modelMats = c.modelMatrices;
    auto & diffuseColors = c.diffuseColors;
    auto & baseReflectivity = c.baseReflectivity;
    auto & roughness = c.roughness;
    auto & metallic = c.metallic;
    auto & buffers = c.buffers;
    buffers.Clear();
    _state.gpuBuffers.push_back(buffers);

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

    //buffers.Bind();
}

void RendererBackend::_ClearInstancedData() {
    // glDeleteBuffers(buffers.size(), &buffers[0]);
    for (auto& buffer: _state.gpuBuffers) buffer.Clear();
    _state.gpuBuffers.clear();
}

void RendererBackend::_BindShader(Pipeline * s) {
    _UnbindShader();
    s->bind();
    _state.currentShader = s;
}

void RendererBackend::_UnbindShader() {
    if (!_state.currentShader) return;
    //_unbindAllTextures();
    _state.currentShader->unbind();
    _state.currentShader = nullptr;
}

static void SetCullState(const RenderFaceCulling & mode) {
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

static bool ValidateTexture(const Async<Texture> & tex) {
    return tex.Completed() && !tex.Failed();
}

void RendererBackend::_Render(const RenderNodeView& e, bool removeViewTranslation) {
    const Camera& camera = *_frame->camera;
    const glm::mat4 & projection = _frame->projection;
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
    _UnbindShader();

    // Set up the shader we will use for this batch of entities
    Pipeline * s;
    std::vector<RendererEntityData>* meshContainer;
    if (e.Get()->GetLightInteractionEnabled() == false) {
        s = _state.forward.get();
        meshContainer = &_frame->instancedFlatMeshes.find(e)->second;
    }
    else {
        s = _state.geometry.get();
        meshContainer = &_frame->instancedPbrMeshes.find(e)->second;
    }

    //s->print();
    _BindShader(s);

    s->setMat4("projection", &projection[0][0]);
    s->setMat4("view", &view[0][0]);

#define SETUP_TEXTURE(name, flag, handle)           \
        tex = _LookupTexture(handle);               \
        const bool valid = ValidateTexture(tex);    \
        s->setBool(flag, valid);                    \
        if (valid) {                                \
            s->bindTexture(name, tex.Get());        \
        }

    for (int i = 0; i < meshContainer->size(); ++i) {
        Async<Texture> tex;
        const RenderMeshContainer* c = e.Get()->GetMeshContainer(i);
        const RendererEntityData& container = (*meshContainer)[i];

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
        SetCullState(e.Get()->GetFaceCullMode());

        c->mesh->Render(container.size, container.buffers);
    }

#undef SETUP_TEXTURE

    _UnbindShader();
}

void RendererBackend::_RenderCSMDepth() {
    _BindShader(_state.csmDepth.get());
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    // Allows GPU to perform angle-dependent depth offset to help reduce artifacts such as shadow acne
    //glEnable(GL_POLYGON_OFFSET_FILL);
    //glPolygonOffset(0.5f, 1.0f);
    //glBlendFunc(GL_ONE, GL_ONE);
    // glDisable(GL_CULL_FACE);

    _state.csmDepth->setVec3("lightDir", &_frame->csc.worldLightCamera->getDirection()[0]);

    // Set up each individual view-projection matrix
    for (int i = 0; i < _frame->csc.cascades.size(); ++i) {
        auto& csm = _frame->csc.cascades[i];
        _state.csmDepth->setMat4("shadowMatrices[" + std::to_string(i) + "]", &csm.projectionViewRender[0][0]);
    }

    // Render everything in a single pass
    _frame->csc.fbo.bind();
    const Texture * depth = _frame->csc.fbo.getDepthStencilAttachment();
    if (!depth) {
        throw std::runtime_error("Critical error: depth attachment not present");
    }
    glViewport(0, 0, depth->width(), depth->height());
    // Render each entity into the depth map
    for (auto& viewMesh : _frame->csc.visible) {
        for (int i = 0; i < viewMesh.second.size(); ++i) {
            const RendererEntityData& container = viewMesh.second[i];
            const RenderNodeView& e = viewMesh.first;
            const RenderMeshPtr m = e.Get()->GetMeshContainer(i)->mesh;
            const size_t numInstances = container.size;
            SetCullState(e.Get()->GetFaceCullMode());
            m->Render(numInstances, container.buffers);
        }
    }
    _frame->csc.fbo.unbind();

    _UnbindShader();
    //glDisable(GL_POLYGON_OFFSET_FILL);
}

void RendererBackend::RenderScene() {
    const Camera& c = *_frame->camera;
    // constexpr size_t maxBytesPerFrame = 1024 * 1024 * 32; // 32 mb per frame
    // size_t totalBytes = 0;

    // std::vector<InstancedData *> instEntities = std::vector<InstancedData *>{
    //     &_frame->instancedPbrMeshes,
    //     &_frame->instancedFlatMeshes  
    // };

    // for (auto instData : instEntities) {
    //     for (auto& entityView : *instData) {
    //         //auto rnode = entityView.first.Get()->GetRenderNode();
    //         auto rnode = entityView.first.Get();
    //         for (int i = 0; i < rnode->GetNumMeshContainers(); ++i) {
    //             if (totalBytes > maxBytesPerFrame) break;

    //             if (rnode->GetMeshContainer(i)->mesh->IsGpuDirty()) {
    //                 rnode->GetMeshContainer(i)->mesh->GenerateGpuData();
    //                 totalBytes += rnode->GetMeshContainer(i)->mesh->GetGpuSizeBytes();
    //                 ResourceManager::Instance()->FinalizeModelMemory(rnode->GetMeshContainer(i)->mesh);
    //             }
    //         }
    //     }
    // }

    const int maxInstances = 250;
    const int maxShadowCastingLights = 8;
    const int maxTotalLights = 256;
    const int maxShadowUpdatesPerFrame = maxShadowCastingLights;

    std::unordered_map<LightPtr, bool> perLightIsDirty;
    std::vector<std::pair<LightPtr, double>> perLightDistToViewer;
    // This one is just for shadow-casting lights
    std::vector<std::pair<LightPtr, double>> perLightShadowCastingDistToViewer;
    // Init per light instance data
    for (auto& entry : _frame->lights) {
        LightPtr light = entry.first;
        auto& lightData = entry.second;
        const double distance = glm::distance(c.getPosition(), light->position);
        perLightDistToViewer.push_back(std::make_pair(light, distance));
        //if (distance > 2 * light->getRadius()) continue;
        perLightIsDirty.insert(std::make_pair(light, lightData.dirty || !_ShadowMapExistsForLight(light)));
        if (light->castsShadows()) {
            perLightShadowCastingDistToViewer.push_back(std::make_pair(light, distance));
        }
    }

    // Sort lights based on distance to viewer
    const auto comparison = [](const std::pair<LightPtr, double> & a, const std::pair<LightPtr, double> & b) {
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

    // Set blend func just for shadow pass
    // glBlendFunc(GL_ONE, GL_ONE);
    glEnable(GL_DEPTH_TEST);
    // Perform the shadow volume pre-pass
    _BindShader(_state.shadows.get());
    int shadowUpdates = 0;
    for (auto&[light, d] : perLightShadowCastingDistToViewer) {
        if (shadowUpdates > maxShadowUpdatesPerFrame) break;
        ++shadowUpdates;
        const double distance = glm::distance(c.getPosition(), light->position);
        // We want to compute shadows at least once for each light source before we enable the option of skipping it 
        // due to it being too far away
        const bool dirty = perLightIsDirty.find(light)->second;
        //if (distance > 2 * light->getRadius() || !dirty) continue;
        if (!dirty) continue;

        auto & instancedMeshes = _frame->lights.find(light)->second.visible;
    
        // TODO: Make this work with spotlights
        //PointLightPtr point = (PointLightPtr)light;
        PointLight * point = (PointLight *)light.get();
        const ShadowMap3D & smap = this->_shadowMap3DHandles.find(_GetShadowMapHandleForLight(light))->second;

        const glm::mat4 lightPerspective = glm::perspective<float>(glm::radians(90.0f), float(smap.shadowCubeMap.width()) / smap.shadowCubeMap.height(), point->getNearPlane(), point->getFarPlane());

        // glBindFramebuffer(GL_FRAMEBUFFER, smap.frameBuffer);
        smap.frameBuffer.clear(glm::vec4(1.0f));
        smap.frameBuffer.bind();
        glViewport(0, 0, smap.shadowCubeMap.width(), smap.shadowCubeMap.height());
        // Current pass only cares about depth buffer
        // glClear(GL_DEPTH_BUFFER_BIT);

        auto transforms = GenerateLightViewTransforms(lightPerspective, point->position);
        for (int i = 0; i < transforms.size(); ++i) {
            const std::string index = "[" + std::to_string(i) + "]";
            _state.shadows->setMat4("shadowMatrices" + index, &transforms[i][0][0]);
        }
        _state.shadows->setVec3("lightPos", &light->position[0]);
        _state.shadows->setFloat("farPlane", point->getFarPlane());

        for (auto & entityObservers : instancedMeshes) {
            for (int i = 0; i < entityObservers.second.size(); ++i) {
                RenderMeshPtr m = entityObservers.first.Get()->GetMeshContainer(i)->mesh;
                SetCullState(entityObservers.first.Get()->GetFaceCullMode());
                m->Render(entityObservers.second[i].size, entityObservers.second[i].buffers);
            }
        }

        // Unbind
        smap.frameBuffer.unbind();
    }
    _UnbindShader();

    // Perform world light depth pass if enabled
    if (_frame->csc.worldLightingEnabled) {
        _RenderCSMDepth();
    }

    // TEMP: Set up the light source
    //glm::vec3 lightPos(0.0f, 0.0f, 0.0f);
    //glm::vec3 lightColor(10.0f); 

    // Make sure to bind our own frame buffer for rendering
    _state.buffer.fbo.bind();
    
    // Make sure some of our global GL states are set properly for primary rendering below
    glBlendFunc(_state.blendSFactor, _state.blendDFactor);
    glViewport(0, 0, _frame->viewportWidth, _frame->viewportHeight);

    // Begin geometry pass
    //glEnable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    for (auto & entityObservers : _frame->instancedPbrMeshes) {
        const RenderNodeView& e = entityObservers.first;
        _Render(e);
    }
    _state.buffer.fbo.unbind();

    glDisable(GL_CULL_FACE);
    //glEnable(GL_BLEND);

    // Begin deferred lighting pass
    _state.lightingFbo.bind();
    glDisable(GL_DEPTH_TEST);
    //_unbindAllTextures();
    _BindShader(_state.lighting.get());
    _InitLights(_state.lighting.get(), perLightDistToViewer, maxShadowCastingLights);
    _state.lighting->bindTexture("gPosition", _state.buffer.position);
    _state.lighting->bindTexture("gNormal", _state.buffer.normals);
    _state.lighting->bindTexture("gAlbedo", _state.buffer.albedo);
    _state.lighting->bindTexture("gBaseReflectivity", _state.buffer.baseReflectivity);
    _state.lighting->bindTexture("gRoughnessMetallicAmbient", _state.buffer.roughnessMetallicAmbient);
    _RenderQuad();
    _state.lightingFbo.unbind();
    _UnbindShader();

    // Forward pass for all objects that don't interact with light (may also be used for transparency later as well)
    _state.lightingFbo.copyFrom(_state.buffer.fbo, BufferBounds{0, 0, _frame->viewportWidth, _frame->viewportHeight}, BufferBounds{0, 0, _frame->viewportWidth, _frame->viewportHeight}, BufferBit::DEPTH_BIT, BufferFilter::NEAREST);
    // Blit to default framebuffer - not that the framebuffer you are writing to has to match the internal format
    // of the framebuffer you are reading to!
    glEnable(GL_DEPTH_TEST);
    _state.lightingFbo.bind();
    for (auto & entityObservers : _frame->instancedFlatMeshes) {
        const RenderNodeView& e = entityObservers.first;
        _Render(e);
    }
    _state.lightingFbo.unbind();
    // glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Enable post-FX effects such as bloom
    _PerformPostFxProcessing();

    // Perform final drawing to screen + gamma correction
    _FinalizeFrame();
}

void RendererBackend::_PerformPostFxProcessing() {
    glDisable(GL_CULL_FACE);
    glDisable(GL_BLEND);
    //glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);

    // We use this so that we can avoid a final copy between the downsample and blurring stages
    std::vector<PostFXBuffer> finalizedPostFxFrames(_state.numDownsampleIterations + _state.numUpsampleIterations);
   
    Pipeline* bloom = _state.bloom.get();
    _BindShader(bloom);

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
        _RenderQuad();
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
            _RenderQuad();
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
        _RenderQuad();
        buffer.fbo.unbind();
        
        finalizedPostFxFrames[postFXIndex] = buffer;
        _state.finalScreenTexture = buffer.fbo.getColorAttachments()[0];
    }

    _UnbindShader();
}

void RendererBackend::_FinalizeFrame() {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDisable(GL_CULL_FACE);
    glViewport(0, 0, _frame->viewportWidth, _frame->viewportHeight);
    //glEnable(GL_BLEND);

    // Now render the screen
    _BindShader(_state.hdrGamma.get());
    _state.hdrGamma->bindTexture("screen", _state.finalScreenTexture);
    _RenderQuad();
    _UnbindShader();
}

void RendererBackend::End() {
    if ( !_frame->vsyncEnabled ) {
        // 0 lets it run as fast as it can
        SDL_GL_SetSwapInterval(0);
    }

    // Swap front and back buffer
    SDL_GL_SwapWindow(_window);    
}

std::vector<SDL_Event> RendererBackend::PollInputEvents() {
    return std::move(_state.events);
}

RendererMouseState RendererBackend::GetMouseState() const {
    return _state.mouse;
}

void RendererBackend::_RenderQuad() {
    _state.screenQuad->GetMeshContainer(0)->mesh->Render(1, GpuArrayBuffer());
}

TextureHandle RendererBackend::CreateShadowMap3D(uint32_t resolutionX, uint32_t resolutionY) {
    ShadowMap3D smap;
    smap.shadowCubeMap = Texture(TextureConfig{TextureType::TEXTURE_3D, TextureComponentFormat::DEPTH, TextureComponentSize::BITS_DEFAULT, TextureComponentType::FLOAT, resolutionX, resolutionY, 0, false}, nullptr);
    smap.shadowCubeMap.setMinMagFilter(TextureMinificationFilter::LINEAR, TextureMagnificationFilter::LINEAR);
    smap.shadowCubeMap.setCoordinateWrapping(TextureCoordinateWrapping::CLAMP_TO_EDGE);
    // We need to set this when using sampler2DShadow in the GLSL shader
    //smap.shadowCubeMap.setTextureCompare(TextureCompareMode::COMPARE_REF_TO_TEXTURE, TextureCompareFunc::LEQUAL);

    smap.frameBuffer = FrameBuffer({smap.shadowCubeMap});
    if (!smap.frameBuffer.valid()) {
        _isValid = false;
        return TextureHandle::Null();
    }
    TextureHandle handle = TextureHandle::NextHandle();
    this->_shadowMap3DHandles.insert(std::make_pair(handle, smap));
    return handle;
}

Async<Texture> RendererBackend::_LookupTexture(TextureHandle handle) const {
    Async<Texture> ret;
    ResourceManager::Instance()->GetTexture(handle, ret);
    return ret;
}

Texture RendererBackend::_LookupShadowmapTexture(TextureHandle handle) const {
    if (handle == TextureHandle::Null()) return Texture();

    if (_shadowMap3DHandles.find(handle) == _shadowMap3DHandles.end()) {
        return Texture();
    }

    return _shadowMap3DHandles.find(handle)->second.shadowCubeMap;
}

void RendererBackend::_InitLights(Pipeline * s, const std::vector<std::pair<LightPtr, double>> & lights, const size_t maxShadowLights) {
    // Set up point lights

    const Camera& c = *_frame->camera;
    glm::vec3 lightColor;
    int lightIndex = 0;
    int shadowLightIndex = 0;
    int i = 0;
    for (; i < lights.size(); ++i) {
        LightPtr light = lights[i].first;
        PointLight * point = (PointLight *)light.get();
        const double distance = lights[i].second; //glm::distance(c.getPosition(), light->position);
        // Skip lights too far from camera
        //if (distance > (2 * light->getRadius())) continue;
        lightColor = point->getBaseColor() * point->getIntensity();
        s->setVec3("lightPositions[" + std::to_string(lightIndex) + "]", &point->position[0]);
        s->setVec3("lightColors[" + std::to_string(lightIndex) + "]", &lightColor[0]);
        s->setFloat("lightRadii[" + std::to_string(lightIndex) + "]", point->getRadius());
        s->setBool("lightCastsShadows[" + std::to_string(lightIndex) + "]", point->castsShadows());
        //_bindShadowMapTexture(s, "shadowCubeMaps[" + std::to_string(lightIndex) + "]", light->getShadowMapHandle());
        if (point->castsShadows() && shadowLightIndex < maxShadowLights) {
            s->setFloat("lightFarPlanes[" + std::to_string(shadowLightIndex) + "]", point->getFarPlane());
            //_bindShadowMapTexture(s, "shadowCubeMaps[" + std::to_string(shadowLightIndex) + "]", _getShadowMapHandleForLight(light));
            s->bindTexture("shadowCubeMaps[" + std::to_string(shadowLightIndex) + "]", _LookupShadowmapTexture(_GetShadowMapHandleForLight(light)));
            ++shadowLightIndex;
        }
        ++lightIndex;
    }

    if (shadowLightIndex == 0) {
       // If we don't do this the fragment shader crashes
       s->setFloat("lightFarPlanes[0]", 0.0f);
       //_bindShadowMapTexture(s, "shadowCubeMaps[0]", _state.dummyCubeMap);
       s->bindTexture("shadowCubeMaps[0]", _LookupShadowmapTexture(_state.dummyCubeMap));
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
    const Camera & lightCam = *_frame->csc.worldLightCamera;
    glm::mat4 lightWorld = lightCam.getWorldTransform();
    // glm::mat4 lightView = lightCam.getViewTransform();
    glm::vec3 direction = lightCam.getDirection(); //glm::vec3(-lightWorld[2].x, -lightWorld[2].y, -lightWorld[2].z);
    // STRATUS_LOG << "Light direction: " << direction << std::endl;
    s->setBool("infiniteLightingEnabled", _frame->csc.worldLightingEnabled);
    s->setVec3("infiniteLightDirection", &direction[0]);
    lightColor = _frame->csc.worldLightColor;
    s->setVec3("infiniteLightColor", &lightColor[0]);

    s->bindTexture("infiniteLightShadowMap", *_frame->csc.fbo.getDepthStencilAttachment());
    for (int i = 0; i < _frame->csc.cascades.size(); ++i) {
        //s->bindTexture("infiniteLightShadowMaps[" + std::to_string(i) + "]", *_state.csms[i].fbo.getDepthStencilAttachment());
        s->setMat4("cascadeProjViews[" + std::to_string(i) + "]", &_frame->csc.cascades[i].projectionViewSample[0][0]);
        // s->setFloat("cascadeSplits[" + std::to_string(i) + "]", _state.cascadeSplits[i]);
    }

    for (int i = 0; i < 2; ++i) {
        s->setVec4("shadowOffset[" + std::to_string(i) + "]", &_frame->csc.cascadeShadowOffsets[i][0]);
    }

    for (int i = 0; i < 3; ++i) {
        // s->setVec3("cascadeScale[" + std::to_string(i) + "]", &_state.csms[i + 1].cascadeScale[0]);
        // s->setVec3("cascadeOffset[" + std::to_string(i) + "]", &_state.csms[i + 1].cascadeOffset[0]);
        s->setVec4("cascadePlanes[" + std::to_string(i) + "]", &_frame->csc.cascades[i + 1].cascadePlane[0]);
    }

    // s->setMat4("cascade0ProjView", &_state.csms[0].projectionView[0][0]);
}

TextureHandle RendererBackend::_GetShadowMapHandleForLight(LightPtr light) {
    assert(_shadowMap3DHandles.size() > 0);

    auto it = _lightsToShadowMap.find(light);
    // If not found, look for an existing shadow map
    if (it == _lightsToShadowMap.end()) {
        TextureHandle handle;
        for (const auto & entry : _shadowMap3DHandles) {
            if (_usedShadowMaps.find(entry.first) == _usedShadowMaps.end()) {
                handle = entry.first;
                break;
            }
        }

        if (handle == TextureHandle::Null()) {
            // Evict oldest since we could not find an available handle
            LightPtr oldest = _lruLightCache.front();
            _lruLightCache.pop_front();
            handle = _lightsToShadowMap.find(oldest)->second;
            _EvictLightFromShadowMapCache(oldest);
        }

        _SetLightShadowMapHandle(light, handle);
        _AddLightToShadowMapCache(light);
        return handle;
    }

    // Update the LRU cache
    _AddLightToShadowMapCache(light);
    return it->second;
}

void RendererBackend::_SetLightShadowMapHandle(LightPtr light, TextureHandle handle) {
    _lightsToShadowMap.insert(std::make_pair(light, handle));
    _usedShadowMaps.insert(handle);
}

void RendererBackend::_EvictLightFromShadowMapCache(LightPtr light) {
    for (auto it = _lruLightCache.begin(); it != _lruLightCache.end(); ++it) {
        if (*it == light) {
            _lruLightCache.erase(it);
            return;
        }
    }
}

bool RendererBackend::_ShadowMapExistsForLight(LightPtr light) {
    return _lightsToShadowMap.find(light) != _lightsToShadowMap.end();
}

void RendererBackend::_AddLightToShadowMapCache(LightPtr light) {
    // First remove the existing light entry if it's already there
    _EvictLightFromShadowMapCache(light);
    // Push to back so that it is seen as most recently used
    _lruLightCache.push_back(light);
}
}