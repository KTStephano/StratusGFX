
#include <includes/Renderer.h>
#include <iostream>
#include "includes/Shader.h"
#include "includes/Renderer.h"

Renderer::Renderer(SDL_Window * window) {
    const int32_t maxGLVersion = 3;
    const int32_t minGLVersion = 2;

    // Set the profile to core as opposed to immediate mode
    SDL_GL_SetAttribute(SDL_GLattr::SDL_GL_CONTEXT_PROFILE_MASK,
            SDL_GL_CONTEXT_PROFILE_CORE);
    // Set max/min version to be 3.2
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, maxGLVersion);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, minGLVersion);
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

    if (!gl3wIsSupported(maxGLVersion, minGLVersion)) {
        std::cerr << "[error] OpenGL 3.2 not supported" << std::endl;
        _isValid = false;
        return;
    }

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

    _isValid = true;

    // Initialize the shaders
    Shader * noLightNoTexture = new Shader("../includes/no_texture_no_lighting.vs",
            "../includes/no_texture_no_lighting.fs");
    _shaders.push_back(noLightNoTexture);
    _propertyShaderMap.insert(std::make_pair(FLAT, noLightNoTexture));
    _isValid = _isValid && noLightNoTexture->isValid();
}

Renderer::~Renderer() {
    if (_context) {
        SDL_GL_DeleteContext(_context);
        _context = nullptr;
    }
    for (Shader * shader : _shaders) delete shader;
    _shaders.clear();
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

const Shader *Renderer::getCurrentShader() const {
    return nullptr;
}
