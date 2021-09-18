
#ifndef STRATUSGFX_RENDERER_H
#define STRATUSGFX_RENDERER_H

#include <string>
#include <vector>
#include <list>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include "StratusEntity.h"
#include "StratusCommon.h"
#include "StratusCamera.h"
#include "StratusRenderNode.h"
#include "StratusTexture.h"
#include "StratusFrameBuffer.h"
#include "StratusLight.h"
#include "StratusMath.h"
#include "StratusGpuBuffer.h"
#include "StratusThread.h"

namespace stratus {
    class Pipeline;
    class Light;
    class InfiniteLight;
    class Quad;
    struct PostProcessFX;

    struct RendererEntityData {
        std::vector<glm::mat4> modelMatrices;
        std::vector<glm::vec3> diffuseColors;
        std::vector<glm::vec3> baseReflectivity;
        std::vector<float> roughness;
        std::vector<float> metallic;
        GpuArrayBuffer buffers;
        size_t size = 0;    
        // if true, regenerate buffers
        bool dirty;
    };

    typedef std::unordered_map<RenderNodeView, std::vector<RendererEntityData>> InstancedData;

    struct RendererLightData {
        InstancedData visible; 
        // If true then its shadow maps should be regenerated
        bool dirty;
    };

    struct RendererCascadeData {
        InstancedData visible;
        // Use during shadow map rendering
        glm::mat4 projectionViewRender;
        // Use during shadow map sampling
        glm::mat4 projectionViewSample;
        glm::vec3 cascadePlane;
        glm::vec3 cascadePosition;
        float cascadeRadius;
        float cascadeBegins;
        float cascadeEnds;
    };

    struct RendererCascadeContainer {
        FrameBuffer fbo;
        std::vector<RendererCascadeData> cascades;
        glm::vec4 cascadeShadowOffsets[2];
        uint32_t cascadeResolutionXY;
        glm::vec3 worldLightColor;
        CameraPtr worldLightCamera;
        bool regenerateFbo;
        bool worldLightingEnabled;
    };

    // Represents data for current active frame
    struct RendererFrame {
        uint32_t viewportWidth;
        uint32_t viewportHeight;
        CameraPtr camera;
        RendererCascadeContainer csc;
        InstancedData instancedPbrMeshes;
        InstancedData instancedFlatMeshes;
        std::unordered_map<LightPtr, RendererLightData> lights;
        glm::mat4 projection;
        glm::vec4 clearColor;
        bool viewportDirty;
        bool vsyncEnabled;
    };

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

    class RendererBackend {
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
            int numShadowMaps = 20;
            int shadowCubeMapX = 1024, shadowCubeMapY = 1024;
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
            // Need to keep track of these to clear them at the end of each frame
            std::vector<GpuArrayBuffer> gpuBuffers;
            // For everything else including bloom post-processing
            int numBlurIterations = 10;
            // Might change from frame to frame
            int numDownsampleIterations = 0;
            int numUpsampleIterations = 0;
            std::vector<PostFXBuffer> gaussianBuffers;
            std::vector<PostFXBuffer> postFxBuffers;
            // End of the pipeline should write to this
            Texture finalScreenTexture;
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
            std::unique_ptr<Pipeline> bloom;
            // Handles cascading shadow map depth buffer rendering
            std::unique_ptr<Pipeline> csmDepth;
            std::vector<Pipeline *> shaders;
            // Generic screen quad so we can render the screen
            // from a separate frame buffer
            RenderNodePtr screenQuad;
            // Gets around what might be a driver bug...
            TextureHandle dummyCubeMap;
            // Window events
            std::vector<SDL_Event> events;
        };

        struct TextureCache {
            std::string file;
            TextureHandle handle = TextureHandle::Null();
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
         * This encodes the same information as the _textures map, except
         * that it can be indexed by a TextureHandle for fast lookup of
         * texture handles attached to Material objects.
         */
        //mutable std::unordered_map<TextureHandle, TextureCache> _textureHandles;

        /**
         * Maps all shadow maps to a handle.
         */
        std::unordered_map<TextureHandle, ShadowMap3D> _shadowMap3DHandles;

        // Lights -> Handles map
        std::unordered_map<LightPtr, TextureHandle> _lightsToShadowMap;

        // Marks which maps are in use by an active light
        std::unordered_set<TextureHandle> _usedShadowMaps;

        // Marks which lights are currently in the cache
        std::list<LightPtr> _lruLightCache;

        // Current frame data used for drawing
        std::shared_ptr<RendererFrame> _frame;

        /**
         * If the renderer was setup properly then this will be marked
         * true.
         */
        bool _isValid = false;

    public:
        explicit RendererBackend(const uint32_t width, const uint32_t height, const std::string&);
        ~RendererBackend();

        /**
         * @return graphics configuration which includes
         *      details about various hardware capabilities
         */
        const GFXConfig & Config() const;

        /**
         * @return true if the renderer initialized itself properly
         *      and false if any errors occurred
         */
        bool Valid() const;

        const Pipeline * GetCurrentShader() const;

        //void invalidateAllTextures();

        void RecompileShaders();

        /**
         * Attempts to load a model if not already loaded. Be sure to check
         * the returned model's isValid() function.
         */
        // Model loadModel(const std::string & file);

        TextureHandle CreateShadowMap3D(uint32_t resolutionX, uint32_t resolutionY);

        /**
         * Sets the render mode to be either ORTHOGRAPHIC (2d)
         * or PERSPECTIVE (3d).
         */
        // void setRenderMode(RenderMode mode);

        /**
         * IMPORTANT! This sets up the renderer for a new frame.
         *
         * @param clearScreen if false then renderer will begin
         * drawing without clearing the screen
         */
        void Begin(const std::shared_ptr<RendererFrame>&, bool clearScreen);

        // Takes all state set during Begin and uses it to render the scene
        void RenderScene();

        /**
         * Finalizes the current scene and displays it.
         */
        void End();

        // Returns window events since the last time this was called
        std::vector<SDL_Event> PollInputEvents();

    private:
        void _ClearGBuffer();
        void _AddDrawable(const EntityPtr& e);
        void _UpdateWindowDimensions();
        void _ClearFramebufferData(const bool);
        void _InitAllInstancedData();
        void _InitLights(Pipeline * s, const std::vector<std::pair<LightPtr, double>> & lights, const size_t maxShadowLights);
        void _InitInstancedData(RendererEntityData &);
        void _ClearInstancedData();
        void _BindShader(Pipeline *);
        void _UnbindShader();
        void _PerformPostFxProcessing();
        void _FinalizeFrame();
        void _InitializePostFxBuffers();
        void _Render(const RenderNodeView &, bool removeViewTranslation = false);
        void _RenderCSMDepth();
        void _RenderQuad();
        TextureHandle _GetShadowMapHandleForLight(LightPtr);
        void _SetLightShadowMapHandle(LightPtr, TextureHandle);
        void _EvictLightFromShadowMapCache(LightPtr);
        void _AddLightToShadowMapCache(LightPtr);
        bool _ShadowMapExistsForLight(LightPtr);
        Async<Texture> _LookupTexture(TextureHandle handle) const;
        Texture _LookupShadowmapTexture(TextureHandle handle) const;
        void _RecalculateCascadeData();
        void _ValidateAllShaders();
    };
}

#endif //STRATUSGFX_RENDERER_H
