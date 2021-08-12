
#include <Renderer.h>
#include <iostream>
#include <Light.h>
#include "Shader.h"
#include "Renderer.h"
#include "Quad.h"
#include "Utils.h"
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

    printGLInfo(_config);
    _isValid = true;

    // Initialize the shaders
    Shader * noLightNoTexture = new Shader("../resources/shaders/no_texture_no_lighting.vs",
            "../resources/shaders/no_texture_no_lighting.fs");
    _shaders.push_back(noLightNoTexture);
    Shader * noLightTexture = new Shader("../resources/shaders/texture_no_lighting.vs",
            "../resources/shaders/texture_no_lighting.fs");
    _shaders.push_back(noLightTexture);
    Shader * lightTexture = new Shader("../resources/shaders/texture_pbr.vs",
            "../resources/shaders/texture_pbr.fs");
    _shaders.push_back(lightTexture);
    Shader * lightTextureNormalMap = new Shader("../resources/shaders/texture_pbr_nm.vs",
                                                "../resources/shaders/texture_pbr_nm.fs");
    _shaders.push_back(lightTextureNormalMap);
    /*
    Shader * lightTextureNormalDepthMap = new Shader("../resources/shaders/texture_lighting_nm_dm.vs",
                                                     "../resources/shaders/texture_lighting_nm_dm.fs");
    */
    Shader * lightTextureNormalDepthMap = new Shader("../resources/shaders/texture_pbr_nm_dm.vs",
                                                     "../resources/shaders/texture_pbr_nm_dm.fs");
    Shader * lightTextureNormalDepthRoughnessMap = new Shader("../resources/shaders/texture_pbr_nm_dm_rm.vs",
                                                     "../resources/shaders/texture_pbr_nm_dm_rm.fs");
    Shader * lightTextureNormalDepthRoughnessEnvironmentMap = new Shader("../resources/shaders/texture_pbr_nm_dm_rm_ao.vs",
                                                     "../resources/shaders/texture_pbr_nm_dm_rm_ao.fs");                             
    _state.pbrShader = lightTextureNormalDepthRoughnessEnvironmentMap;          
    _shaders.push_back(lightTextureNormalDepthMap);
    using namespace std;
    // Now we need to insert the shaders into the property map - this allows
    // the renderer to perform quick lookup to determine the shader that matches
    // all of a RenderEntities rendering requirements

    // Insert flat-lighting shaders
    std::unordered_map<uint32_t, Shader *> shaderMap;
    shaderMap.insert(make_pair(NONE, noLightNoTexture));
    shaderMap.insert(make_pair(TEXTURED, noLightTexture));
    _propertyShaderMap.insert(std::make_pair(FLAT, shaderMap));

    // Insert dynamic-lighting shaders
    shaderMap.clear();
    shaderMap.insert(make_pair(TEXTURED, lightTexture));
    shaderMap.insert(make_pair(TEXTURED | NORMAL_MAPPED, lightTextureNormalMap));
    shaderMap.insert(make_pair(TEXTURED | NORMAL_HEIGHT_MAPPED, lightTextureNormalDepthMap));
    shaderMap.insert(make_pair(TEXTURED | NORMAL_HEIGHT_MAPPED | ROUGHNESS_MAPPED, lightTextureNormalDepthRoughnessMap));
    shaderMap.insert(make_pair(TEXTURED | NORMAL_HEIGHT_MAPPED | ROUGHNESS_MAPPED | ENVIRONMENT_MAPPED, lightTextureNormalDepthRoughnessEnvironmentMap));
    _propertyShaderMap.insert(std::make_pair(DYNAMIC, shaderMap));
    
    // Now we need to establish a mapping between all of the possible render
    // property combinations with a list of entities that match those requirements
    _state.entities.insert(make_pair(FLAT, vector<RenderEntity *>()));
    _state.entities.insert(make_pair(DYNAMIC, vector<RenderEntity *>()));

    // Set up the hdr/gamma postprocessing shader
    _state.hdrGamma = std::make_unique<Shader>("../resources/shaders/hdr.vs",
            "../resources/shaders/hdr.fs");

    // Set up the shadow preprocessing shader
    _state.shadows = std::make_unique<Shader>("../resources/shaders/shadow.vs",
       "../resources/shaders/shadow.gs",
       "../resources/shaders/shadow.fs");

    // Create the screen quad
    _state.screenQuad = std::make_unique<Quad>();

    // Use the shader isValid() method to determine if everything succeeded
    _isValid = _isValid &&
            noLightNoTexture->isValid() &&
            noLightTexture->isValid() &&
            lightTexture->isValid() &&
            lightTextureNormalMap->isValid() &&
            lightTextureNormalDepthMap->isValid() &&
            lightTextureNormalDepthRoughnessMap->isValid() &&
            _state.hdrGamma->isValid() &&
            _state.shadows->isValid();
}

Renderer::~Renderer() {
    if (_context) {
        SDL_GL_DeleteContext(_context);
        _context = nullptr;
    }
    for (Shader * shader : _shaders) delete shader;
    _shaders.clear();
    invalidateAllTextures();

    // Delete the main frame buffer
    glDeleteFramebuffers(1, &_state.frameBuffer);
    glDeleteTextures(1, &_state.colorBuffer);
    glDeleteTextures(1, &_state.depthBuffer);
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
    std::cout << _state.fov
        << " " << _state.znear
        << " " << _state.zfar
        << " " << _state.windowWidth
        << " " << _state.windowHeight << std::endl;
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

    // Regenerate the main frame buffer
    glDeleteFramebuffers(1, &_state.frameBuffer);
    glDeleteTextures(1, &_state.colorBuffer);
    glDeleteTextures(1, &_state.depthBuffer);

    glGenFramebuffers(1, &_state.frameBuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, _state.frameBuffer);

    // Create the color buffer - notice that is uses higher
    // than normal precision. This allows us to write color values
    // greater than 1.0 to support things like HDR.
    glGenTextures(1, &_state.colorBuffer);
    glBindTexture(GL_TEXTURE_2D, _state.colorBuffer);
    glTexImage2D(GL_TEXTURE_2D, // target
            0, // level
            GL_RGB16F, // internal format
            _state.windowWidth, // width
            _state.windowHeight, // height
            0, // border
            GL_RGBA, // format
            GL_FLOAT, // type
            nullptr); // data
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // Attach the color buffer to the frame buffer
    glFramebufferTexture2D(GL_FRAMEBUFFER,
            GL_COLOR_ATTACHMENT0,
            GL_TEXTURE_2D,
            _state.colorBuffer,
            0);

    // Create the depth buffer
    glGenTextures(1, &_state.depthBuffer);
    glBindTexture(GL_TEXTURE_2D, _state.depthBuffer);
    glTexImage2D(GL_TEXTURE_2D,
            0,
            GL_DEPTH_COMPONENT,
            _state.windowWidth,
            _state.windowHeight,
            0,
            GL_DEPTH_COMPONENT,
            GL_FLOAT,
            nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // Attach the depth buffer to the frame buffer
    glFramebufferTexture2D(GL_FRAMEBUFFER,
            GL_DEPTH_ATTACHMENT,
            GL_TEXTURE_2D,
            _state.depthBuffer,
            0);

    // Check the status to make sure it's complete
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "[error] Generating frame buffer failed" << std::endl;
        _isValid = false;
        return;
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void Renderer::setPerspectiveData(float fov, float fnear, float ffar) {
    // TODO: Find the best lower bound for fov instead of arbitrary 25.0f
    if (fov < 25.0f) return;
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
        glBindFramebuffer(GL_FRAMEBUFFER, _state.frameBuffer);
        glClearColor(_state.clearColor.r, _state.clearColor.g,
                     _state.clearColor.b, _state.clearColor.a);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
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

    glEnable(GL_BLEND);
    //glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_CULL_FACE);
    glFrontFace(GL_CW);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LINE_SMOOTH);
    glEnable(GL_POLYGON_SMOOTH);

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
        _entitiesSeenBefore.insert(std::make_pair(e, EntityStateInfo{e->position, true}));
    }
    else {
        EntityStateInfo & info = _entitiesSeenBefore.find(e)->second;
        const double distance = glm::distance(e->position, info.lastPos);
        if (distance > 0.25) {
            info.lastPos = e->position;
            info.dirty = true;
        }
        else {
            info.dirty = false;
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
        //                       pos       pos + dir                               up
        projection * glm::lookAt(lightPos, lightPos + glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
        projection * glm::lookAt(lightPos, lightPos + glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
        projection * glm::lookAt(lightPos, lightPos + glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
        projection * glm::lookAt(lightPos, lightPos + glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f)),
        projection * glm::lookAt(lightPos, lightPos + glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
        projection * glm::lookAt(lightPos, 
        lightPos + glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, -1.0f, 0.0f))
    };
}

void Renderer::_initInstancedData(__MeshContainer & c, std::vector<GLuint> & buffers) {
    Shader * pbr = _state.pbrShader;

    auto & modelMats = c.modelMatrices;
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

void Renderer::_bindShader(Shader * s) {
    s->bind();
    _state.currentShader = s;
}

void Renderer::_unbindShader() {
    if (!_state.currentShader) return;
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

void Renderer::end(const Camera & c) {
    // Pull the view transform/projection matrices
    const glm::mat4 * projection = &_state.perspective;
    const glm::mat4 * view = &c.getViewTransform();
    const int maxInstances = 250;
    const int maxShadowCastingLights = 11;
    // Need to delete these at the end of the frame
    std::vector<GLuint> buffers;

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
        if (distance > light->getRadius()) continue;
        perLightInstancedMeshes.insert(std::make_pair(light, std::unordered_map<__RenderEntityObserver, std::unordered_map<__MeshObserver, __MeshContainer>>()));
        perLightIsDirty.insert(std::make_pair(light, _lightsSeenBefore.find(light)->second.dirty));
        perLightShadowCastingDistToViewer.push_back(std::make_pair(light, distance));
    }

    // Sort lights based on distance to viewer
    const auto comparison = [](const std::pair<Light *, double> & a, const std::pair<Light *, double> & b) {
        return a.second < b.second;
    };
    std::sort(perLightDistToViewer.begin(), perLightDistToViewer.end(), comparison);
    std::sort(perLightShadowCastingDistToViewer.begin(), perLightShadowCastingDistToViewer.end(), comparison);

    // Remove lights that exceed our max count
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
    glBlendFunc(GL_ONE, GL_ONE);
    // Perform the shadow volume pre-pass
    _bindShader(_state.shadows.get());
    for (auto&[light, d] : perLightShadowCastingDistToViewer) {
        const double distance = glm::distance(c.getPosition(), light->position);
        // We want to compute shadows at least once for each light source before we enable the option of skipping it 
        // due to it being too far away
        const bool dirty = perLightIsDirty.find(light)->second;
        if (distance > light->getRadius() || !dirty) continue;
        
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
        const ShadowMap3D & smap = this->_shadowMap3DHandles.find(point->getShadowMapHandle())->second;

        const glm::mat4 lightPerspective = glm::perspective<float>(glm::radians(90.0f), smap.width / smap.height, point->getNearPlane(), point->getFarPlane());

        glBindFramebuffer(GL_FRAMEBUFFER, smap.frameBuffer);
        glViewport(0, 0, smap.width, smap.height);
        // Current pass only cares about depth buffer
        glClear(GL_DEPTH_BUFFER_BIT);

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
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        _clearInstancedData(buffers);
    }
    _clearInstancedData(buffers);
    _unbindAllTextures();
    _unbindShader();

    // Init the instance data which enables us to drastically reduce the number of draw calls
    for (auto & entityObservers : _state.instancedMeshes) {
        for (auto & meshObservers : entityObservers.second) {
            _initInstancedData(meshObservers.second, buffers);
        }
    }

    // TEMP: Set up the light source
    //glm::vec3 lightPos(0.0f, 0.0f, 0.0f);
    //glm::vec3 lightColor(10.0f); 

    // Make sure to bind our own frame buffer for rendering
    glBindFramebuffer(GL_FRAMEBUFFER, _state.frameBuffer);
    
    // Make sure some of our global GL states are set properly for primary rendering below
    glBlendFunc(_state.blendSFactor, _state.blendDFactor);
    glViewport(0, 0, _state.windowWidth, _state.windowHeight);

//    for (auto & p : _state.entities) {
    for (auto & entityObservers : _state.instancedMeshes) {
        for (auto & meshObservers : entityObservers.second) {
            //_initInstancedData(meshObservers.second, buffers);
            RenderEntity * e = entityObservers.first.e;
            Mesh * m = meshObservers.first.m;
            const size_t numInstances = meshObservers.second.size;

            // Unbind current shader if one is bound
            _unbindShader();

            // Set up the shader we will use for this batch of entities
            uint32_t lightProperties = e->getLightProperties();
            uint32_t renderProperties = m->getRenderProperties();
            if (_propertyShaderMap.find(lightProperties) == _propertyShaderMap.end()) {
                std::cout << "Error! Unable to find map with given light properties" << std::endl;
                continue;
            }
            auto outer = _propertyShaderMap.find(lightProperties)->second;
            if (outer.find(renderProperties) == outer.end()) {
                std::cout << "Error! Unable to find shader with given render properties" << std::endl;
                continue;
            }
            auto inner = _propertyShaderMap.find(lightProperties)->second.find(renderProperties);
            Shader * s = inner->second;
            //s->print();
            _bindShader(s);

            // Set up uniforms specific to this type of shader
            //if (properties == (DYNAMIC | TEXTURED) || properties == (DYNAMIC | TEXTURED | NORMAL_MAPPED)) {
            const float dynTextured = (lightProperties & DYNAMIC);
            const float normOrHeight = (renderProperties & TEXTURED) || (renderProperties & NORMAL_MAPPED) || (renderProperties & NORMAL_HEIGHT_MAPPED);
            const bool lightingEnabled = dynTextured || (dynTextured && normOrHeight);
            /**
            if (dynTextured || (dynTextured && normOrHeight)) {
                glm::vec3 lightColor;
                for (int i = 0; i < _state.lights.size(); ++i) {
                    Light * light = _state.lights[i];
                    lightColor = light->getColor() * light->getIntensity();
                    s->setVec3("lightPositions[" + std::to_string(i) + "]",
                            &light->position[0]);
                    s->setVec3("lightColors[" + std::to_string(i) + "]",
                            &lightColor[0]);
                }
                s->setInt("numLights", (int)_state.lights.size());
                s->setVec3("viewPosition", &c.getPosition()[0]);
                s->setMat4("view", &(*view)[0][0]);
            }
            */

            /*
            if (_state.mode == RenderMode::ORTHOGRAPHIC) {
                projection = &_state.orthographic;
            } else {
                projection = &_state.perspective;
            }
            */
            s->setMat4("projection", &(*projection)[0][0]);
            s->setMat4("view", &(*view)[0][0]);

            // Iterate through all entities and draw them
            //s->setMat4("projection", &(*projection)[0][0]);
            const glm::mat4 & model = e->model;

            if (lightingEnabled) _initLights(s, c, perLightDistToViewer, perLightShadowCastingDistToViewer);

            if (renderProperties & TEXTURED) {
                /*
                glActiveTexture(GL_TEXTURE0);
                s->setInt("diffuseTexture", 0);
                GLuint texture = _lookupTexture(e->getMaterial().texture);
                glBindTexture(GL_TEXTURE_2D + 0, texture);
                */
                _bindTexture(s, "diffuseTexture", m->getMaterial().texture);
            }

            // Determine which uniforms we should set
            if (lightProperties & FLAT) {
                s->setVec3("diffuseColor", &m->getMaterial().diffuseColor[0]);
                //glm::mat4 modelView = (*view) * model;
                //s->setMat4("modelView", &modelView[0][0]);
            } else if (lightProperties & DYNAMIC) {
                //s->setFloat("shininess", e->getMaterial().specularShininess);
                //s->setMat4("model", &model[0][0]);
                if (renderProperties & NORMAL_MAPPED || renderProperties & NORMAL_HEIGHT_MAPPED) {
                    /*
                    s->setInt("normalMap", 0);
                    glActiveTexture(GL_TEXTURE0 + 0);
                    GLuint normalMap = _lookupTexture(e->getMaterial().normalMap);
                    glBindTexture(GL_TEXTURE_2D + 0, normalMap);
                    */
                    _bindTexture(s, "normalMap", m->getMaterial().normalMap);
                }

                if (renderProperties & NORMAL_HEIGHT_MAPPED) {
                    _bindTexture(s, "depthMap", m->getMaterial().depthMap);
                    s->setFloat("heightScale", m->getMaterial().heightScale);
                }

                if (renderProperties & ROUGHNESS_MAPPED) {
                    _bindTexture(s, "roughnessMap", m->getMaterial().roughnessMap);
                }

                if (renderProperties & ENVIRONMENT_MAPPED) {
                    _bindTexture(s, "ambientOcclusionMap", m->getMaterial().environmentMap);
                }
            }

            setCullState(m->cullingMode);

            m->bind();
            m->render(numInstances);
            m->unbind();
            _unbindAllTextures();
            //glBindTexture(GL_TEXTURE_2D, 0);

            //_clearInstancedData(buffers);
            _unbindShader();
        }
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Now render the screen
    glDisable(GL_CULL_FACE);
    _bindShader(_state.hdrGamma.get());
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, _state.colorBuffer);
    _state.hdrGamma->setInt("screen", 0);
    _state.screenQuad->bind();
    _state.screenQuad->render(1);
    _state.screenQuad->unbind();
    glBindTexture(GL_TEXTURE_2D, 0);
    _unbindShader();

    // Make sure to clear out all instanced data used this frame
    _clearInstancedData(buffers);
}

TextureHandle Renderer::loadTexture(const std::string &file) {
    auto it = _textures.find(file);
    if (it != _textures.end()) return it->second.handle;

    Texture2D tex;
    int width, height, numChannels;
    // @see http://www.redbancosdealimentos.org/homes-flooring-design-sources
    uint8_t * data = stbi_load(file.c_str(), &width, &height, &numChannels, 0);
    if (data) {
        glGenTextures(1, &tex.texture);
        tex.handle = int(_textures.size() + 1);
        GLenum internalFormat;
        GLenum dataFormat;
        // This loads the textures with sRGB in mind so that they get converted back
        // to linear color space. Warning: if the texture was not actually specified as an
        // sRGB texture (common for normal/specular maps), this will cause problems.
        switch (numChannels) {
            case 1:
                internalFormat = GL_RED;
                dataFormat = GL_RED;
                break;
            case 3:
                internalFormat = GL_SRGB;
                dataFormat = GL_RGB;
                break;
            case 4:
                internalFormat = GL_SRGB_ALPHA;
                dataFormat = GL_RGBA;
                break;
            default:
                std::cerr << "[error] Unknown texture loading error -"
                    << " format may be invalid" << std::endl;
                glDeleteTextures(1, &tex.texture);
                stbi_image_free(data);
                return -1;
        }

        glBindTexture(GL_TEXTURE_2D, tex.texture);
        glTexImage2D(GL_TEXTURE_2D,
                0,
                internalFormat,
                width, height,
                0,
                dataFormat,
                GL_UNSIGNED_BYTE,
                data);
        glGenerateMipmap(GL_TEXTURE_2D);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        // Do not use MIPMAP_LINEAR here as it does not make sense with magnification
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glBindTexture(GL_TEXTURE_2D, 0);
    } else {
        std::cerr << "[error] Could not load texture: " << file << std::endl;
        return -1;
    }
    _textures.insert(std::make_pair(file, tex));
    _textureHandles.insert(std::make_pair(tex.handle, tex));
    stbi_image_free(data);
    return tex.handle;
}

Model Renderer::loadModel(const std::string & file) {
    auto it = this->_models.find(file);
    if (it != this->_models.end()) {
        return it->second;
    }

    Model m(*this, file);
    this->_models.insert(std::make_pair(file, m));
    return std::move(m);
}

ShadowMapHandle Renderer::createShadowMap3D(int resolutionX, int resolutionY) {
    ShadowMap3D smap;
    smap.width = resolutionX;
    smap.height = resolutionY;
    glGenFramebuffers(1, &smap.frameBuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, smap.frameBuffer);
    // Generate the 3D depth buffer
    glGenTextures(1, &smap.shadowCubeMap);
    glBindTexture(GL_TEXTURE_CUBE_MAP, smap.shadowCubeMap);
    for (int face = 0; face < 6; ++face) {
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + face, 0, GL_DEPTH_COMPONENT, resolutionX,
                     resolutionY, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    }
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE); // Notice the third dimension
    glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, smap.shadowCubeMap, 0);
    // Tell OpenGL we won't be using a color buffer
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);
    // Unbind framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    TextureHandle handle = (int)(this->_shadowMap3DHandles.size() + 1);
    this->_shadowMap3DHandles.insert(std::make_pair(handle, smap));
    return handle;
}

void Renderer::invalidateAllTextures() {
    for (auto & texture : _textures) {
        glDeleteTextures(1, &texture.second.texture);
        // Make sure we mark it as unloaded just in case someone tries
        // to use it in the future
        texture.second.loaded = false;
    }
}

GLuint Renderer::_lookupTexture(TextureHandle handle) const {
    auto it = _textureHandles.find(handle);
    // TODO: Make sure that 0 actually signifies an invalid texture in OpenGL
    if (it == _textureHandles.end()) return 0;
    return it->second.texture;
}

// TODO: Need a way to clean up point light resources
void Renderer::addPointLight(Light * light) {
    assert(light->getType() == LightType::POINTLIGHT || light->getType() == LightType::SPOTLIGHT);
    _state.lights.push_back(light);

    if (light->getType() == LightType::POINTLIGHT) {
        PointLight * point = (PointLight *)light;
        if (point->getShadowMapHandle() == -1) {
            point->_setShadowMapHandle(this->createShadowMap3D(1024, 1024));
        }
    }

    if (_lightsSeenBefore.find(light) == _lightsSeenBefore.end()) {
        _lightsSeenBefore.insert(std::make_pair(light, EntityStateInfo{light->position, true}));
    }
    else {
        EntityStateInfo & info = _lightsSeenBefore.find(light)->second;
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

void Renderer::_bindTexture(Shader * s, const std::string & textureName, TextureHandle handle) {
    GLuint texture = _lookupTexture(handle);
    int textureIndex = (int)_state.boundTextures.size();
    glActiveTexture(GL_TEXTURE0 + textureIndex);
    s->setInt(textureName, textureIndex);
    glBindTexture(GL_TEXTURE_2D, texture);
    _state.boundTextures.insert(std::make_pair(textureIndex, BoundTextureInfo{textureIndex, texture, TextureType::TEXTURE_2D, textureName}));
}

void Renderer::_bindShadowMapTexture(Shader * s, const std::string & textureName, ShadowMapHandle handle) {
    const ShadowMap3D & smap = _shadowMap3DHandles.find(handle)->second;
    int textureIndex = (int)_state.boundTextures.size();
    glActiveTexture(GL_TEXTURE0 + textureIndex);
    s->setInt(textureName, textureIndex);
    glBindTexture(GL_TEXTURE_CUBE_MAP, smap.shadowCubeMap);
    _state.boundTextures.insert(std::make_pair(textureIndex, BoundTextureInfo{textureIndex, smap.shadowCubeMap, TextureType::TEXTURE_CUBE_MAP, textureName}));
}

void Renderer::_unbindAllTextures() {
    /**
    for (size_t i = 0; i < _state.boundTextures.size(); ++i) {
        int textureIndex = (int)i;
        glActiveTexture(GL_TEXTURE0 + i);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
    */
   for (auto & e : _state.boundTextures) {
       int textureIndex = e.second.textureIndex;
       GLuint texture = e.second.texture;
       TextureType type = e.second.type;
       const std::string & name = e.second.name;
       glActiveTexture(GL_TEXTURE0 + textureIndex);
       if (_state.currentShader) {
           _state.currentShader->setInt(name, 0);
       }
       if (type == TextureType::TEXTURE_2D) {
           glBindTexture(GL_TEXTURE_2D, 0);
       }
       else {
           glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
       }
   }
    _state.boundTextures.clear();
}

void Renderer::_initLights(Shader * s, const Camera & c, const std::vector<std::pair<Light *, double>> & lights, const std::vector<std::pair<Light *, double>> & shadowLights) {
    glm::vec3 lightColor;
    int lightIndex = 0;
    int shadowLightIndex = 0;
    for (int i = 0; i < lights.size(); ++i) {
        PointLight * light = (PointLight *)lights[i].first;
        const double distance = lights[i].second; //glm::distance(c.getPosition(), light->position);
        // Skip lights too far from camera
        if (distance > (2 * light->getRadius())) continue;
        lightColor = light->getColor() * light->getIntensity();
        s->setVec3("lightPositions[" + std::to_string(lightIndex) + "]", &light->position[0]);
        s->setVec3("lightColors[" + std::to_string(lightIndex) + "]", &lightColor[0]);
        s->setFloat("lightFarPlanes[" + std::to_string(lightIndex) + "]", light->getFarPlane());
        //_bindShadowMapTexture(s, "shadowCubeMaps[" + std::to_string(lightIndex) + "]", light->getShadowMapHandle());
        if (lightIndex < shadowLights.size()) {
            _bindShadowMapTexture(s, "shadowCubeMaps[" + std::to_string(lightIndex) + "]", light->getShadowMapHandle());
            ++shadowLightIndex;
        }
        ++lightIndex;
    }
    s->setInt("numLights", lightIndex);
    s->setInt("numShadowLights", shadowLightIndex);
    s->setVec3("viewPosition", &c.getPosition()[0]);
}
}