
#ifndef STRATUSGFX_RENDERER_H
#define STRATUSGFX_RENDERER_H

#include <string>
#include "Common.h"

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
     * If the renderer was setup properly then this will be marked
     * true.
     */
    bool _isValid = false;

public:
    Renderer(SDL_Window * window);
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
};

#endif //STRATUSGFX_RENDERER_H
