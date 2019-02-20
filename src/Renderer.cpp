
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
    using namespace std;
    _propertyShaderMap.insert(make_pair(FLAT, noLightNoTexture));
    _state.entities.insert(make_pair(FLAT, vector<shared_ptr<RenderEntity>>()));
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

void Renderer::_recalculateProjMatrices() {
    _state.perspective = glm::perspective(glm::radians(_state.fov),
            float(_state.windowWidth) / float(_state.windowHeight),
            _state.znear,
            _state.zfar);
    // arguments: left, right, bottom, top, near, far - this matrix
    // transforms [0,width] to [-1, 1] and [0, height] to [-1, 1]
    _state.orthographic = glm::ortho(0.0f, float(_state.windowWidth),
            float(_state.windowHeight), 0.0f, -1.0f, 1.0f);
}

void Renderer::_setWindowDimensions(int w, int h) {
    if (_state.windowWidth == w && _state.windowHeight == h) return;
    if (w < 0 || h < 0) return;
    _state.windowWidth = w;
    _state.windowHeight = h;
    _recalculateProjMatrices();
    glViewport(0, 0, w, h);
}

void Renderer::setPerspectiveData(float fov, float near, float far) {
    // TODO: Find the best lower bound for fov instead of arbitrary 25.0f
    if (fov < 25.0f) return;
    _state.fov = fov;
    _state.znear = near;
    _state.zfar = far;
    _recalculateProjMatrices();
}

void Renderer::setRenderMode(RenderMode mode) {
    _state.mode = mode;
}

void Renderer::begin(bool clearScreen) {
    // Make sure we set our context as the active one
    SDL_GL_MakeCurrent(_window, _context);

    if (clearScreen) {
        glClearColor(_state.clearColor.r, _state.clearColor.g,
                _state.clearColor.b, _state.clearColor.a);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }

    // Check for changes in the window size
    int w, h;
    SDL_GetWindowSize(_window, &w, &h);
    // This won't resize anything if the width/height
    // didn't change
    _setWindowDimensions(w, h);

    // Clear all entities from the previous frame
    for (auto & e : _state.entities) {
        e.second.clear();
    }
}

void Renderer::addDrawable(std::shared_ptr<RenderEntity> e) {
    auto it = _state.entities.find(e->getRenderProperties());
    if (it == _state.entities.end()) {
        // Not necessarily an error since if an entity is set to
        // invisible, we won't bother adding them
        //std::cerr << "[error] Unable to add entity" << std::endl;
        return;
    }
    it->second.push_back(e);
}

static void inset(glm::mat4 & out, const glm::mat3 & in) {
    out[0].x = in[0].x;
    out[0].y = in[0].y;
    out[0].z = in[0].z;

    out[1].x = in[1].x;
    out[1].y = in[1].y;
    out[1].z = in[1].z;

    out[2].x = in[2].x;
    out[2].y = in[2].y;
    out[2].z = in[2].z;
}

static void setScale(glm::mat4 & out, const glm::vec3 & scale) {
    out[0].x = out[0].x * scale.x;
    out[0].y = out[0].y * scale.y;
    out[0].z = out[0].z * scale.z;

    out[1].x = out[1].x * scale.x;
    out[1].y = out[1].y * scale.y;
    out[1].z = out[1].z * scale.z;

    out[2].x = out[2].x * scale.x;
    out[2].y = out[2].y * scale.y;
    out[2].z = out[2].z * scale.z;
}

static void setTranslate(glm::mat4 & out, const glm::vec3 & translate) {
    out[3].x = translate.x;
    out[3].y = translate.y;
    out[3].z = translate.z;
}

void Renderer::end(std::shared_ptr<Camera> c) {
    if (c == nullptr) {
        std::cerr << "[error] begin() called with null camera" << std::endl;
        return;
    }
    for (auto & p : _state.entities) {
        // Set up the shader we will use for this batch of entities
        if (_state.currentShader != nullptr) {
            _state.currentShader->unbind();
        }
        uint32_t properties = p.first;
        auto it = _propertyShaderMap.find(properties);
        Shader * s = it->second;
        _state.currentShader = s;
        it->second->bind();

        // Pull the view transform/projection matrices
        const glm::mat4 * projection;
        const glm::mat4 * view = &c->getViewTransform();
        glm::mat4 model(1.0f);
        if (_state.mode == RenderMode::ORTHOGRAPHIC) {
            projection = &_state.orthographic;
        } else {
            projection = &_state.perspective;
        }
        s->setMat4("projection", &(*projection)[0][0]);

        // Iterate through all entities and draw them
        for (auto & e : p.second) {
            //s->setMat4("projection", &(*projection)[0][0]);
            inset(model, e->rotation);
            setScale(model, e->scale);
            setTranslate(model, e->position);
            glm::mat4 modelView = (*view) * model;
            s->setMat4("modelView", &modelView[0][0]);

            // Determine which uniforms we should set
            if (properties & FLAT) {
                s->setVec3("diffuseColor", &e->getMaterial().diffuseColor[0]);
            }
            e->render();
        }
    }
}
