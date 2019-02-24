
#include <includes/Renderer.h>
#include <iostream>
#include <includes/Light.h>
#include "includes/Shader.h"
#include "includes/Renderer.h"
#include "includes/Quad.h"
#define STB_IMAGE_IMPLEMENTATION
#include "includes/STBImage.h"

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

    _isValid = true;

    // Initialize the shaders
    Shader * noLightNoTexture = new Shader("../resources/shaders/no_texture_no_lighting.vs",
            "../resources/shaders/no_texture_no_lighting.fs");
    _shaders.push_back(noLightNoTexture);
    Shader * noLightTexture = new Shader("../resources/shaders/texture_no_lighting.vs",
            "../resources/shaders/texture_no_lighting.fs");
    _shaders.push_back(noLightTexture);
    Shader * lightTexture = new Shader("../resources/shaders/texture_lighting.vs",
            "../resources/shaders/texture_lighting.fs");
    _shaders.push_back(lightTexture);
    using namespace std;
    // Now we need to insert the shaders into the property map - this allows
    // the renderer to perform quick lookup to determine the shader that matches
    // all of a RenderEntities rendering requirements
    _propertyShaderMap.insert(make_pair(FLAT, noLightNoTexture));
    _propertyShaderMap.insert(make_pair(FLAT | TEXTURED, noLightTexture));
    _propertyShaderMap.insert(make_pair(DYNAMIC | TEXTURED, lightTexture));
    // Now we need to establish a mapping between all of the possible render
    // property combinations with a list of entities that match those requirements
    _state.entities.insert(make_pair(FLAT, vector<RenderEntity *>()));
    _state.entities.insert(make_pair(FLAT | TEXTURED, vector<RenderEntity *>()));
    _state.entities.insert(make_pair(DYNAMIC | TEXTURED, vector<RenderEntity *>()));

    // Set up the hdr/gamma preprocessing shader
    _state.hdrGamma = std::make_unique<Shader>("../resources/shaders/hdr.vs",
            "../resources/shaders/hdr.fs");

    // Create the screen quad
    _state.screenQuad = std::make_unique<Quad>();

    // Use the shader isValid() method to determine if everything succeeded
    _isValid = _isValid &&
            noLightNoTexture->isValid() &&
            noLightTexture->isValid() &&
            lightTexture->isValid() &&
            _state.hdrGamma->isValid();
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

    // Clear all lights
    _state.lights.clear();

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
    auto it = _state.entities.find(e->getRenderProperties());
    if (it == _state.entities.end()) {
        // Not necessarily an error since if an entity is set to
        // invisible, we won't bother adding them
        std::cerr << "[error] Unable to add entity" << std::endl;
        return;
    }
    it->second.push_back(e);
}

static void rotate(glm::mat4 & out, const glm::vec3 & angles) {
    float angleX = glm::radians(angles.x);
    float angleY = glm::radians(angles.y);
    float angleZ = glm::radians(angles.z);

    float cx = std::cos(angleX);
    float cy = std::cos(angleY);
    float cz = std::cos(angleZ);

    float sx = std::sin(angleX);
    float sy = std::sin(angleY);
    float sz = std::sin(angleZ);

    out[0] = glm::vec4(cy * cz,
                       sx * sy * cz + cx * sz,
                       -cx * sy * cz + sx * sz,
                       out[0].w);

    out[1] = glm::vec4(-cy * sz,
                       -sx * sy * sz + cx * cz,
                       cx * sy * sz + sx * cz,
                       out[1].w);

    out[2] = glm::vec4(sy,
                       -sx * cy,
                       cx * cy, out[2].w);
}

// Inserts a 3x3 matrix into the upper section of a 4x4 matrix
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

static void scale(glm::mat4 & out, const glm::vec3 & scale) {
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

static void translate(glm::mat4 & out, const glm::vec3 & translate) {
    out[3].x = translate.x;
    out[3].y = translate.y;
    out[3].z = translate.z;
}

void Renderer::end(const Camera & c) {
    // Pull the view transform/projection matrices
    const glm::mat4 * projection = &_state.perspective;
    const glm::mat4 * view = &c.getViewTransform();

    // TEMP: Set up the light source
    //glm::vec3 lightPos(0.0f, 0.0f, 0.0f);
    //glm::vec3 lightColor(10.0f);

    // Make sure to bind our own frame buffer for rendering
    glBindFramebuffer(GL_FRAMEBUFFER, _state.frameBuffer);

    for (auto & p : _state.entities) {
        // Set up the shader we will use for this batch of entities
        if (_state.currentShader != nullptr) {
            _state.currentShader->unbind();
        }
        uint32_t properties = p.first;
        auto it = _propertyShaderMap.find(properties);
        Shader * s = it->second;
        _state.currentShader = s;
        s->bind();

        // Set up uniforms specific to this type of shader
        if (properties == (DYNAMIC | TEXTURED)) {
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

        /*
        if (_state.mode == RenderMode::ORTHOGRAPHIC) {
            projection = &_state.orthographic;
        } else {
            projection = &_state.perspective;
        }
         */
        s->setMat4("projection", &(*projection)[0][0]);

        // Iterate through all entities and draw them
        for (auto & e : p.second) {
            //s->setMat4("projection", &(*projection)[0][0]);
            glm::mat4 model(1.0f);
            rotate(model, e->rotation);
            scale(model, e->scale);
            translate(model, e->position);

            // Determine which uniforms we should set
            if (properties & FLAT) {
                s->setVec3("diffuseColor", &e->getMaterial().diffuseColor[0]);
                glm::mat4 modelView = (*view) * model;
                s->setMat4("modelView", &modelView[0][0]);
            } else if (properties & DYNAMIC) {
                s->setFloat("shininess", e->getMaterial().specularShininess);
                s->setMat4("model", &model[0][0]);
            }

            if (properties & TEXTURED) {
                glActiveTexture(GL_TEXTURE0);
                s->setInt("diffuseTexture", 0);
                GLuint texture = _lookupTexture(e->getMaterial().texture);
                glBindTexture(GL_TEXTURE_2D, texture);
            }
            glFrontFace(GL_CW);
            e->render();
            glBindTexture(GL_TEXTURE_2D, 0);
        }

        s->unbind();
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Now render the screen
    _state.hdrGamma->bind();
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, _state.colorBuffer);
    _state.hdrGamma->setInt("screen", 0);
    _state.screenQuad->render();
    glBindTexture(GL_TEXTURE_2D, 0);
    _state.hdrGamma->unbind();
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
        tex.handle = _textures.size() + 1;
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

void Renderer::addPointLight(Light *light) {
    _state.lights.push_back(light);
}
