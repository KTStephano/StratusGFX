
#ifndef STRATUSGFX_RENDERER_H
#define STRATUSGFX_RENDERER_H

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include "Common.h"
#include "RenderEntity.h"
#include "Camera.h"

class Shader;

/**
 * This contains information about a lot of the
 * OpenGL configuration params after initialization
 * takes place.
 */
struct GFXConfig {
    std::string renderer;
    std::string version;
    int32_t maxDrawBuffers;
    int32_t maxCombinedTextures;
    int32_t maxCubeMapTextureSize;
    int32_t maxFragmentUniformVectors;
    int32_t maxFragmentUniformComponents;
    int32_t maxVaryingFloats;
    int32_t maxRenderbufferSize;
    int32_t maxTextureImageUnits;
    int32_t maxTextureSize;
    int32_t maxVertexAttribs;
    int32_t maxVertexUniformVectors;
    int32_t maxVertexUniformComponents;
    int32_t maxViewportDims[2];
};

/**
 * Contains rgba information.
 */
struct Color {
    float r, g, b, a;

    explicit Color(float r = 1.0f, float g = 1.0f,
            float b = 1.0f, float a = 1.0f)
            : r(r), g(g), b(b), a(a) {}
};

class Renderer {
    struct RenderState {
        Color clearColor;
        RenderMode mode = RenderMode::PERSPECTIVE;
        std::unordered_map<uint32_t, std::vector<RenderEntity *>> entities;
        int windowWidth = 0;
        int windowHeight = 0;
        float fov = 90.0f, znear = 0.25f, zfar = 1000.0f;
        glm::mat4 orthographic;
        glm::mat4 perspective;
        //std::shared_ptr<Camera> camera;
        Shader * currentShader;
    };

    struct Texture2D {
        std::string file;
        TextureHandle handle = -1;
        GLuint texture;
        /**
         * If true then the file is currently loaded into memory.
         * If false then it has been unloaded, so if anyone tries
         * to use it then it needs to first be re-loaded.
         */
        bool loaded = true;
    };

    /**
     * This is needed to create the gl context and to
     * perform a gl context switch. This pointer is
     * NOT managed by this class and should not be deleted
     * by the Renderer.
     */
    SDL_Window * _window;

    /**
     * The rendering context is defined as the window +
     * gl context. Together they allow the renderer to
     * perform a context switch before drawing in the event
     * that multiple render objects are being used at once.
     */
    SDL_GLContext _context;

    /**
     * Contains information about various different settings
     * which will affect final rendering.
     */
    RenderState _state;

    /**
     * All the fields in this struct are set during initialization
     * since we have to set up the context and then query OpenGL.
     */
    GFXConfig _config;

    /**
     * Contains all of the shaders that are used by the renderer.
     */
    std::vector<Shader *> _shaders;

    /**
     * This maps a set of properties to a shader. This should be
     * a valid combination, such as FLAT | TEXTURED.
     */
    std::unordered_map<uint32_t, Shader *> _propertyShaderMap;

    /**
     * Contains a list of textures that have been loaded into memory.
     */
    std::unordered_map<std::string, Texture2D> _textures;

    /**
     * This encodes the same information as the _textures map, except
     * that it can be indexed by a TextureHandle for fast lookup of
     * texture handles attached to Material objects.
     */
    std::unordered_map<TextureHandle, Texture2D> _textureHandles;

    /**
     * If the renderer was setup properly then this will be marked
     * true.
     */
    bool _isValid = false;

public:
    explicit Renderer(SDL_Window * window);
    ~Renderer();

    /**
     * @return graphics configuration which includes
     *      details about various hardware capabilities
     */
    const GFXConfig & config() const;

    /**
     * @return true if the renderer initialized itself properly
     *      and false if any errors occurred
     */
     bool valid() const;

     /**
      * Sets the clear color for screen refreshes.
      * @param c clear color
      */
     void setClearColor(const Color & c);

     const Shader * getCurrentShader() const;

     void invalidateAllTextures();

     /**
      * Attempts to load a texture if it hasn't already been loaded.
      * In the event that it was previously loaded previously, it will
      * return an existing handle rather than re-loading the data.
      * @param texture
      * @return texture handle of valid or -1 if invalid
      */
     TextureHandle loadTexture(const std::string & file);

     /**
      * Sets up the arguments for the perspective projection,
      * if/when the render mode is set to PERSPECTIVE.
      * @param fov field of view in degrees
      * @param near near clipping plane (ex: 0.25f)
      * @param far far clipping plane (ex: 1000.0f)
      */
     void setPerspectiveData(float fov, float near, float far);

     /**
      * Sets the render mode to be either ORTHOGRAPHIC (2d)
      * or PERSPECTIVE (3d).
      */
     void setRenderMode(RenderMode mode);

     /**
      * IMPORTANT! This sets up the renderer for a new frame.
      *
      * @param clearScreen if false then renderer will begin
      * drawing without clearing the screen
      */
     void begin(bool clearScreen);

     /**
      * For the current scene, this will add a render entity
      * that is means to be drawn.
      */
     void addDrawable(RenderEntity * e);

     /**
      * Sets the camera for the current scene which will be
      * the camera whose perspective we render from.
      */
     //void setCamera(std::shared_ptr<Camera> c);

     /**
      * Finalizes the current scene and draws it.
      */
     void end(const Camera & c);

private:
    void _setWindowDimensions(int w, int h);
    void _recalculateProjMatrices();

public:
    GLuint _lookupTexture(TextureHandle handle) const;
};

#endif //STRATUSGFX_RENDERER_H
