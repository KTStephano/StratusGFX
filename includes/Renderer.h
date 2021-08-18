
#ifndef STRATUSGFX_RENDERER_H
#define STRATUSGFX_RENDERER_H

#include <string>
#include <vector>
#include <list>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include "Common.h"
#include "RenderEntity.h"
#include "Camera.h"
#include "Model.h"
#include "Texture.h"
#include "FrameBuffer.h"

namespace stratus {
class Pipeline;
class Light;
class Quad;

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

struct __RenderEntityObserver {
    RenderEntity * e;

    __RenderEntityObserver(RenderEntity * e) : e(e) {}

    bool operator==(const __RenderEntityObserver & c) const;
    size_t hashCode() const;
};

struct __MeshObserver {
    Mesh * m;

    __MeshObserver(Mesh * m) : m(m) {}

    bool operator==(const __MeshObserver & c) const;
    size_t hashCode() const;
};

struct __MeshContainer {
    Mesh * m;
    std::vector<glm::mat4> modelMatrices;
    std::vector<glm::vec3> diffuseColors;
    std::vector<glm::vec3> baseReflectivity;
    std::vector<float> roughness;
    std::vector<float> metallic;
    size_t size = 0;

    __MeshContainer(Mesh * m) : m(m) {}
};
}

namespace std {
    template<>
    struct hash<stratus::__RenderEntityObserver> {
        size_t operator()(const stratus::__RenderEntityObserver & c) const {
            return c.hashCode();
        }
    };
}

namespace std {
    template<>
    struct hash<stratus::__MeshObserver> {
        size_t operator()(const stratus::__MeshObserver & c) const {
            return c.hashCode();
        }
    };
}

namespace stratus {
class Renderer {
    enum RTextureType {
        TEXTURE_2D,
        TEXTURE_CUBE_MAP
    };

    struct BoundTextureInfo {
        int textureIndex;
        GLuint texture;
        RTextureType type;
        std::string name;
    };

    struct EntityStateInfo {
        glm::vec3 lastPos;
        glm::vec3 lastScale;
        glm::vec3 lastRotation;
        bool dirty; // if dirty, recompute all shadows
    };

    struct GBuffer {
        FrameBuffer fbo;
        Texture position;                 // RGBA32F (rgba instead of rgb due to possible alignment issues)
        Texture normals;                  // RGBA32F
        Texture albedo;                   // RGBA32F
        Texture baseReflectivity;         // RGBA32F
        Texture roughnessMetallicAmbient; // RGBA32F
        Texture depth;                    // R16F
    };

    struct PostFXBuffer {
        FrameBuffer fbo;
    };

    struct RenderState {
        Color clearColor;
        RenderMode mode = RenderMode::PERSPECTIVE;
        std::unordered_map<uint32_t, std::vector<RenderEntity *>> entities;
        std::vector<RenderEntity *> lightInteractingEntities;
        std::unordered_map<__RenderEntityObserver, std::unordered_map<__MeshObserver, __MeshContainer>> instancedMeshes;
        // These are either point or spotlights and will attenuate with
        // distance
        std::vector<Light *> lights;
        uint32_t windowWidth = 0;
        uint32_t windowHeight = 0;
        float fov = 90.0f, znear = 0.25f, zfar = 1000.0f;
        int numShadowMaps = 10;
        int shadowCubeMapX = 2048, shadowCubeMapY = 2048;
        glm::mat4 orthographic;
        glm::mat4 perspective;
        //std::shared_ptr<Camera> camera;
        Pipeline * currentShader = nullptr;
        // Buffer where all color data is written
        GBuffer buffer;
        // Buffer for lighting pass
        FrameBuffer lightingFbo;
        Texture lightingColorBuffer;
        // Used for effects like bloom
        Texture lightingHighBrightnessBuffer;
        Texture lightingDepthBuffer;
        // For everything else including bloom post-processing
        int numBlurIterations = 10;
        // Might change from frame to frame
        Texture finalBloomColorBuffer;
        std::vector<PostFXBuffer> postFxBuffers;
        // Used for a call to glBlendFunc
        GLenum blendSFactor = GL_ONE;
        GLenum blendDFactor = GL_ZERO;
        // Postprocessing shader which allows for application
        // of hdr and gamma correction
        std::unique_ptr<Pipeline> hdrGamma;
        // Preprocessing shader which sets up the scene to allow for dynamic shadows
        std::unique_ptr<Pipeline> shadows;
        // Geometry pass - handles all combinations of material properties
        std::unique_ptr<Pipeline> geometry;
        // Forward rendering pass
        std::unique_ptr<Pipeline> forward;
        // Handles the lighting stage
        std::unique_ptr<Pipeline> lighting;
        // Stage 1 handles extracting only the bright parts of the scene
        std::unique_ptr<Pipeline> bloomStageOne;
        // Stage 2 takes the bright parts and applies a 9x9 kernel Gaussian Blur several times
        std::unique_ptr<Pipeline> bloomStageTwo;
        // Generic screen quad so we can render the screen
        // from a separate frame buffer
        std::unique_ptr<Quad> screenQuad;
        // Gets around what might be a driver bug...
        TextureHandle dummyCubeMap;
    };

    struct TextureCache {
        std::string file;
        TextureHandle handle = -1;
        Texture texture;
        /**
         * If true then the file is currently loaded into memory.
         * If false then it has been unloaded, so if anyone tries
         * to use it then it needs to first be re-loaded.
         */
        bool loaded = true;
    };

    struct ShadowMap3D {
        // Each shadow map is rendered to a frame buffer backed by a 3D texture
        FrameBuffer frameBuffer;
        Texture shadowCubeMap;
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
    std::vector<Pipeline *> _shaders;

    /**
     * This maps a set of properties to a shader. This should be
     * a valid combination, such as FLAT | TEXTURED.
     */
    //std::unordered_map<uint32_t, std::unordered_map<uint32_t, Pipeline *>> _propertyShaderMap;

    /**
     * Contains a list of textures that have been loaded into memory.
     */
    mutable std::unordered_map<std::string, TextureCache> _textures;

    /**
     * Set of lights added at some point during any frame.
     */
    std::unordered_map<Light *, EntityStateInfo> _lightsSeenBefore;

    /**
     * Set of entities added at some point during any frame.
     */
    std::unordered_map<RenderEntity *, EntityStateInfo> _entitiesSeenBefore;

    /**
     * This encodes the same information as the _textures map, except
     * that it can be indexed by a TextureHandle for fast lookup of
     * texture handles attached to Material objects.
     */
    mutable std::unordered_map<TextureHandle, TextureCache> _textureHandles;

    // Next available handle
    int _nextTextureHandle = 1;

    /**
     * Maps all shadow maps to a handle.
     */
    std::unordered_map<ShadowMapHandle, ShadowMap3D> _shadowMap3DHandles;

    // Lights -> Handles map
    std::unordered_map<Light *, ShadowMapHandle> _lightsToShadowMap;

    // Marks which maps are in use by an active light
    std::unordered_set<ShadowMapHandle> _usedShadowMaps;

    // Marks which lights are currently in the cache
    std::list<Light *> _lruLightCache;

    /**
     * Contains all loaded models indexed by model name.
     */
    std::unordered_map<std::string, Model> _models;

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

     const Pipeline * getCurrentShader() const;

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
      * Attempts to load a model if not already loaded. Be sure to check
      * the returned model's isValid() function.
      */
     Model loadModel(const std::string & file);

     ShadowMapHandle createShadowMap3D(uint32_t resolutionX, uint32_t resolutionY);

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
      * Adds a light to the scene. These lights are considered
      * to be finite in power and so their contribution to the
      * scene will decrease with distance.
      */
     void addPointLight(Light * light);

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
    void _clearGBuffer();
    void _addDrawable(RenderEntity * e, const glm::mat4 &);
    void _setWindowDimensions(int w, int h);
    void _recalculateProjMatrices();
    void _initLights(Pipeline * s, const Camera & c, 
                     const std::vector<std::pair<Light *, double>> & lights, const size_t maxShadowLights);
    void _initInstancedData(__MeshContainer & c, std::vector<GLuint> & buffers);
    void _clearInstancedData(std::vector<GLuint> & buffers);
    void _bindShader(Pipeline *);
    void _unbindShader();
    void _performPostFxProcessing();
    void _finalizeFrame();
    void _initializePostFxBuffers();
    void _buildEntityList(const Camera & c);
    void _render(const Camera &, const RenderEntity *, const Mesh *, const size_t numInstances);
    void _renderQuad();
    ShadowMapHandle _getShadowMapHandleForLight(Light *);
    void _setLightShadowMapHandle(Light *, ShadowMapHandle);
    void _evictLightFromShadowMapCache(Light*);
    void _addLightToShadowMapCache(Light*);
    Texture _lookupTexture(TextureHandle handle) const;
};
}

#endif //STRATUSGFX_RENDERER_H
