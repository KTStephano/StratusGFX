
#ifndef STRATUSGFX_RENDERER_H
#define STRATUSGFX_RENDERER_H

#include <string>
#include <vector>
#include <list>
#include <deque>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include "StratusEntity.h"
#include "StratusCommon.h"
#include "StratusCamera.h"
#include "StratusTexture.h"
#include "StratusFrameBuffer.h"
#include "StratusLight.h"
#include "StratusMath.h"
#include "StratusGpuBuffer.h"
#include "StratusGpuCommon.h"
#include "StratusThread.h"
#include "StratusAsync.h"
#include "StratusEntityCommon.h"
#include "StratusEntity.h"
#include "StratusTransformComponent.h"
#include "StratusRenderComponents.h"
#include "StratusGpuMaterialBuffer.h"
#include "StratusGpuCommandBuffer.h"
#include <functional>
#include "StratusStackAllocator.h"
#include <set>
#include "StratusGpuCommandBuffer.h"

namespace stratus {
    class Pipeline;
    class Light;
    class InfiniteLight;
    class Quad;
    struct PostProcessFX;

    typedef std::function<GpuBuffer(GpuCommandBufferPtr&)> CommandBufferSelectionFunction;

    extern bool IsRenderable(const EntityPtr&);
    extern bool IsLightInteracting(const EntityPtr&);
    extern size_t GetMeshCount(const EntityPtr&);

    enum class RendererCascadeResolution : int {
        CASCADE_RESOLUTION_1024 = 1024,
        CASCADE_RESOLUTION_2048 = 2048,
        CASCADE_RESOLUTION_4096 = 4096,
        CASCADE_RESOLUTION_8192 = 8192
    };

    struct RenderMeshContainer {
        RenderComponent * render = nullptr;
        MeshWorldTransforms * transform = nullptr;
        size_t meshIndex = 0;
    };

    typedef std::shared_ptr<RenderMeshContainer> RenderMeshContainerPtr;

    // struct RendererEntityData {
    //     std::vector<glm::mat4> modelMatrices;
    //     std::vector<glm::vec3> diffuseColors;
    //     std::vector<glm::vec3> baseReflectivity;
    //     std::vector<float> roughness;
    //     std::vector<float> metallic;
    //     GpuArrayBuffer buffers;
    //     size_t size = 0;    
    //     // if true, regenerate buffers
    //     bool dirty;
    // };

    struct RendererMouseState {
        int32_t x;
        int32_t y;
        uint32_t mask;
    };

    typedef std::unordered_map<EntityPtr, std::vector<RenderMeshContainerPtr>> EntityMeshData;

    struct RendererCascadeData {
        GpuCommandReceiveManagerPtr drawCommands;
        // Use during shadow map rendering
        glm::mat4 projectionViewRender;
        // Use during shadow map sampling
        glm::mat4 projectionViewSample;
        // Transforms from a cascade 0 sampling coordinate to the current cascade
        glm::mat4 sampleCascade0ToCurrent;
        glm::vec4 cascadePlane;
        glm::vec3 cascadePositionLightSpace;
        glm::vec3 cascadePositionCameraSpace;
        float cascadeRadius;
        float cascadeBegins;
        float cascadeEnds;
    };

    struct RendererCascadeContainer {
        FrameBuffer fbo;
        std::vector<RendererCascadeData> cascades;
        glm::vec4 cascadeShadowOffsets[2];
        uint32_t cascadeResolutionXY;
        InfiniteLightPtr worldLight;
        CameraPtr worldLightCamera;
        glm::vec3 worldLightDirectionCameraSpace;
        float znear;
        float zfar;
        bool regenerateFbo;    
    };

    struct LightUpdateQueue {
        template<typename LightPtrContainer>
        void PushBackAll(const LightPtrContainer& container) {
            for (const LightPtr& ptr : container) {
                PushBack(ptr);
            }
        }

        void PushFront(const LightPtr& ptr) {
            auto existing = existing_.find(ptr);
            // Allow lights further in the queue to be bumped higher up
            // with this function
            if (existing != existing_.end()) {
                queue_.erase(existing->second);
                existing_.erase(ptr);
            }

            queue_.push_front(ptr);
            auto it = queue_.begin();
            existing_.insert(std::make_pair(ptr, it));
        }

        void PushBack(const LightPtr& ptr) {
            // If a light is already in the queue, don't reorder since it would
            // result in the light losing its better place in line
            if (existing_.find(ptr) != existing_.end() || !ptr->CastsShadows()) return;

            queue_.push_back(ptr);
            auto it = queue_.end();
            --it;
            existing_.insert(std::make_pair(ptr, it));
        }

        LightPtr PopFront() {
            if (Size() == 0) return nullptr;
            auto front = Front();
            existing_.erase(front);
            queue_.pop_front();
            return front;
        }

        LightPtr Front() const {
            if (Size() == 0) return nullptr;
            return queue_.front();
        }

        // In case a light needs to be removed without being updated
        void Erase(const LightPtr& ptr) {
            if (existing_.find(ptr) == existing_.end()) return;
            existing_.erase(ptr);
            for (auto it = queue_.begin(); it != queue_.end(); ++it) {
                const LightPtr& light = *it;
                if (ptr == light) {
                    queue_.erase(it);
                    return;
                }
            }
        }

        // In case all lights need to be removed without being updated
        void Clear() {
            queue_.clear();
            existing_.clear();
        }

        size_t Size() const {
            return queue_.size();
        }

    private:
        std::list<LightPtr> queue_;
        std::unordered_map<LightPtr, std::list<LightPtr>::iterator> existing_;
    };

    // Settings which can be changed at runtime by the application
    struct RendererSettings {
        // These are values we don't need to range check
        TextureHandle skybox = TextureHandle::Null();
        bool vsyncEnabled = false;
        bool globalIlluminationEnabled = true;
        bool fxaaEnabled = true;
        bool taaEnabled = true;
        bool bloomEnabled = true;
        bool usePerceptualRoughness = true;
        RendererCascadeResolution cascadeResolution = RendererCascadeResolution::CASCADE_RESOLUTION_1024;
        // Records how much temporary memory the renderer is allowed to use
        // per frame
        size_t perFrameMaxScratchMemoryBytes = 134217728; // 128 mb

        float GetEmissionStrength() const {
            return emissionStrength_;
        }

        void SetEmissionStrength(const float strength) {
            emissionStrength_ = std::max<float>(strength, 0.0f);
        }

        float GetEmissiveTextureMultiplier() const {
            return emissiveTextureMultiplier_;
        }

        void SetEmissiveTextureMultiplier(const float multiplier) {
            emissiveTextureMultiplier_ = std::max<float>(multiplier, 0.0f);
        }

        glm::vec3 GetFogColor() const {
            return fogColor_;
        }

        float GetFogDensity() const {
            return fogDensity_;
        }

        void SetFogColor(const glm::vec3& color) {
            fogColor_ = glm::vec3(
                std::max<float>(color[0], 0.0f),
                std::max<float>(color[1], 0.0f),
                std::max<float>(color[2], 0.0f)
            );
        }

        void SetFogDensity(const float density) {
            fogDensity_ = std::max<float>(density, 0.0f);
        }

        glm::vec3 GetSkyboxColorMask() const {
            return skyboxColorMask_;
        }

        float GetSkyboxIntensity() const {
            return skyboxIntensity_;
        }

        void SetSkyboxColorMask(const glm::vec3& mask) {
            skyboxColorMask_ = glm::vec3(
                std::max<float>(mask[0], 0.0f),
                std::max<float>(mask[1], 0.0f),
                std::max<float>(mask[2], 0.0f)
            );
        }

        void SetSkyboxIntensity(const float intensity) {
            skyboxIntensity_ = std::max<float>(intensity, 0.0f);
        }

        float GetMinRoughness() const {
            return minRoughness_;
        }

        void SetMinRoughness(const float roughness) {
            minRoughness_ = std::max<float>(roughness, 0.0f);
        }

        void SetAlphaDepthTestThreshold(const float threshold) {
            alphaDepthTestThreshold_ = std::clamp<float>(threshold, 0.0f, 1.0f);
        }

        float GetAlphaDepthTestThreshold() const {
            return alphaDepthTestThreshold_;
        }

        // Values of 1.0 mean that GI occlusion will result in harsh shadow cutoffs
        // Values < 1.0 effectively brighten the scene
        void SetMinGiOcclusionFactor(const float value) {
            minGiOcclusionFactor_ = std::clamp<float>(value, 0.0f, 1.0f);
        }

        float GetMinGiOcclusionFactor() const {
            return minGiOcclusionFactor_;
        }

    private:
        // These are all values we need to range check when they are set
        glm::vec3 fogColor_ = glm::vec3(0.5f);
        float fogDensity_ = 0.0f;
        float emissionStrength_ = 0.0f;
        glm::vec3 skyboxColorMask_ = glm::vec3(1.0f);
        float skyboxIntensity_ = 3.0f;
        float minRoughness_ = 0.08f;
        float alphaDepthTestThreshold_ = 0.5f;
        // This works as a multiplicative effect on top of emission strength
        float emissiveTextureMultiplier_ = 1.0f;
        float minGiOcclusionFactor_ = 0.95f;
    };

    // Represents data for current active frame
    struct RendererFrame {
        uint32_t viewportWidth;
        uint32_t viewportHeight;
        Radians fovy;
        CameraPtr camera;
        std::vector<glm::vec4, StackBasedPoolAllocator<glm::vec4>> viewFrustumPlanes;
        GpuMaterialBufferPtr materialInfo;
        RendererCascadeContainer csc;
        GpuCommandManagerPtr drawCommands;
        std::unordered_set<LightPtr> lights;
        std::unordered_set<LightPtr> virtualPointLights; // data is in lights
        LightUpdateQueue lightsToUpdate; // shadow map data is invalid
        std::unordered_set<LightPtr> lightsToRemove;
        float znear;
        float zfar;
        glm::mat4 projection;
        glm::mat4 view;
        glm::mat4 projectionView;
        glm::mat4 jitterProjectionView;
        glm::mat4 invProjectionView;
        glm::mat4 prevProjectionView = glm::mat4(1.0f);
        glm::mat4 prevInvProjectionView = glm::mat4(1.0f);
        glm::vec4 clearColor;
        RendererSettings settings;
        UnsafePtr<StackAllocator> perFrameScratchMemory;
        bool viewportDirty;
    };

    class RendererBackend {
        // Geometry buffer
        struct GBuffer {
            FrameBuffer fbo;
            //Texture position;                 // RGB16F (rgba instead of rgb due to possible alignment issues)
            Texture normals;                  // RGB16F
            Texture albedo;                   // RGB8F
            Texture baseReflectivity;         // RGB8F
            Texture roughnessMetallicAmbient; // RGB8F
            Texture structure;                // RGBA16F
            Texture velocity;
            Texture id;
            Texture depth;                    // Default bit depth
        };

        struct PostFXBuffer {
            FrameBuffer fbo;
        };

        struct VirtualPointLightData {
            // For splitting viewport into tiles
            const int tileXDivisor = 5;
            const int tileYDivisor = 5;
            // This needs to match what is in the vpl tiled deferred shader compute header!
            int vplShadowCubeMapX = 32, vplShadowCubeMapY = 32;
            //GpuBuffer vplDiffuseMaps;
            //GpuBuffer vplShadowMaps;
            GpuBuffer shadowDiffuseIndices;
            GpuBuffer vplStage1Results;
            GpuBuffer vplVisiblePerTile;
            GpuBuffer vplData;
            GpuBuffer vplUpdatedData;
            GpuBuffer vplVisibleIndices;
            //GpuBuffer vplNumVisible;
            FrameBuffer vplGIFbo;
            FrameBuffer vplGIDenoisedPrevFrameFbo;
            FrameBuffer vplGIDenoisedFbo1;
            FrameBuffer vplGIDenoisedFbo2;
        };

        struct RenderState {
            int numRegularShadowMaps = 200;
            int shadowCubeMapX = 256, shadowCubeMapY = 256;
            int maxShadowCastingLightsPerFrame = 200; // per frame
            int maxTotalRegularLightsPerFrame = 200; // per frame
            GpuBuffer nonShadowCastingPointLights;
            //GpuBuffer shadowCubeMaps;
            GpuBuffer shadowIndices;
            GpuBuffer shadowCastingPointLights;
            VirtualPointLightData vpls;
            // How many shadow maps can be rebuilt each frame
            // Lights are inserted into a queue to prevent any light from being
            // over updated or neglected
            int maxShadowUpdatesPerFrame = 5;
            //std::shared_ptr<Camera> camera;
            Pipeline * currentShader = nullptr;
            // Buffer where all color data is written
            GBuffer currentFrame;
            GBuffer previousFrame;
            // Buffer for lighting pass
            FrameBuffer lightingFbo;
            Texture lightingColorBuffer;
            // Used for effects like bloom
            //Texture lightingHighBrightnessBuffer;
            Texture lightingDepthBuffer;
            FrameBuffer flatPassFboCurrentFrame;
            FrameBuffer flatPassFboPreviousFrame;
            // Used for Screen Space Ambient Occlusion (SSAO)
            Texture ssaoOffsetLookup;               // 4x4 table where each pixel is (16-bit, 16-bit)
            Texture ssaoOcclusionTexture;
            FrameBuffer ssaoOcclusionBuffer;        // Contains light factors computed per pixel
            Texture ssaoOcclusionBlurredTexture;
            FrameBuffer ssaoOcclusionBlurredBuffer; // Counteracts low sample count of occlusion buffer by depth-aware blurring
            // Used for atmospheric shadowing
            FrameBuffer atmosphericFbo;
            Texture atmosphericTexture;
            Texture atmosphericNoiseTexture;
            // Used for gamma-tonemapping
            PostFXBuffer gammaTonemapFbo;
            // Used for fast approximate anti-aliasing (FXAA)
            PostFXBuffer fxaaFbo1;
            PostFXBuffer fxaaFbo2;
            // Used for temporal anti-aliasing (TAA)
            PostFXBuffer taaFbo;
            // Need to keep track of these to clear them at the end of each frame
            std::vector<GpuArrayBuffer> gpuBuffers;
            // For everything else including bloom post-processing
            int numBlurIterations = 10;
            // Might change from frame to frame
            int numDownsampleIterations = 0;
            int numUpsampleIterations = 0;
            std::vector<PostFXBuffer> gaussianBuffers;
            std::vector<PostFXBuffer> postFxBuffers;
            // Handles atmospheric post processing
            PostFXBuffer atmosphericPostFxBuffer;
            // End of the pipeline should write to this
            FrameBuffer finalScreenBuffer;
            // Used for TAA
            FrameBuffer previousFrameBuffer;
            // Used for a call to glBlendFunc
            GLenum blendSFactor = GL_ONE;
            GLenum blendDFactor = GL_ZERO;
            // Depth prepass
            std::unique_ptr<Pipeline> depthPrepass;
            // Skybox
            std::unique_ptr<Pipeline> skybox;
            std::unique_ptr<Pipeline> skyboxLayered;
            // Postprocessing shader which allows for application
            // of hdr and gamma correction
            std::unique_ptr<Pipeline> gammaTonemap;
            // Preprocessing shader which sets up the scene to allow for dynamic shadows
            std::unique_ptr<Pipeline> shadows;
            std::unique_ptr<Pipeline> vplShadows;
            // Geometry pass - handles all combinations of material properties
            std::unique_ptr<Pipeline> geometry;
            // Forward rendering pass
            std::unique_ptr<Pipeline> forward;
            // Handles first SSAO pass (no blurring)
            std::unique_ptr<Pipeline> ssaoOcclude;
            // Handles second SSAO pass (blurring)
            std::unique_ptr<Pipeline> ssaoBlur;
            // Handles the atmospheric shadowing stage
            std::unique_ptr<Pipeline> atmospheric;
            // Handles atmospheric post fx stage
            std::unique_ptr<Pipeline> atmosphericPostFx;
            // Handles the lighting stage
            std::unique_ptr<Pipeline> lighting;
            std::unique_ptr<Pipeline> lightingWithInfiniteLight;
            // Handles global illuminatino stage
            std::unique_ptr<Pipeline> vplGlobalIllumination;
            std::unique_ptr<Pipeline> vplGlobalIlluminationDenoising;
            // Bloom stage
            std::unique_ptr<Pipeline> bloom;
            // Handles virtual point light culling
            std::unique_ptr<Pipeline> vplCulling;
            std::unique_ptr<Pipeline> vplColoring;
            std::unique_ptr<Pipeline> vplTileDeferredCullingStage1;
            std::unique_ptr<Pipeline> vplTileDeferredCullingStage2;
            // Draws axis-aligned bounding boxes
            std::unique_ptr<Pipeline> aabbDraw;
            // Handles cascading shadow map depth buffer rendering
            // (we compile one depth shader per cascade - max 6)
            std::vector<std::unique_ptr<Pipeline>> csmDepth;
            std::vector<std::unique_ptr<Pipeline>> csmDepthRunAlphaTest;
            // Handles fxaa luminance followed by fxaa smoothing
            std::unique_ptr<Pipeline> fxaaLuminance;
            std::unique_ptr<Pipeline> fxaaSmoothing;
            // Handles temporal anti-aliasing
            std::unique_ptr<Pipeline> taa;
            // Performs full screen pass through
            std::unique_ptr<Pipeline> fullscreen;
            std::vector<Pipeline *> shaders;
            // Generic unit cube to render as skybox
            EntityPtr skyboxCube;
            // Generic screen quad so we can render the screen
            // from a separate frame buffer
            EntityPtr screenQuad;
            // Gets around what might be a driver bug...
            TextureHandle dummyCubeMap;
            std::vector<GpuCommandReceiveManagerPtr> dynamicPerPointLightDrawCalls;
            std::vector<GpuCommandReceiveManagerPtr> staticPerPointLightDrawCalls;
            std::unique_ptr<Pipeline> viscullPointLights;
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

        struct ShadowMapCache {
            // Framebuffer which wraps around all available cube maps
            std::vector<FrameBuffer> buffers;

            // Lights -> Handles map
            std::unordered_map<LightPtr, GpuAtlasEntry> lightsToShadowMap;

            // Lists all shadow maps which are currently available
            std::list<GpuAtlasEntry> freeShadowMaps;

            // Marks which lights are currently in the cache
            std::list<LightPtr> cachedLights;
        };

        // Contains the cache for regular lights
        ShadowMapCache smapCache_;

        // Contains the cache for virtual point lights
        ShadowMapCache vplSmapCache_;

        /**
         * Contains information about various different settings
         * which will affect final rendering.
         */
        RenderState state_;

        /**
         * Contains all of the shaders that are used by the renderer.
         */
        std::vector<Pipeline *> shaders_;

        /**
         * This encodes the same information as the _textures map, except
         * that it can be indexed by a TextureHandle for fast lookup of
         * texture handles attached to Material objects.
         */
        //mutable std::unordered_map<TextureHandle, TextureCache> _textureHandles;

        // Current frame data used for drawing
        std::shared_ptr<RendererFrame> frame_;

        // Contains some number of Halton sequence values
        GpuBuffer haltonSequence_;

        /**
         * If the renderer was setup properly then this will be marked
         * true.
         */
        bool isValid_ = false;

    public:
        explicit RendererBackend(const uint32_t width, const uint32_t height, const std::string&);
        ~RendererBackend();

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
        void RenderScene(const double);

        /**
         * Finalizes the current scene and displays it.
         */
        void End();

        // Returns window events since the last time this was called
        // std::vector<SDL_Event> PollInputEvents();

        // // Returns the mouse status as of the most recent frame
        // RendererMouseState GetMouseState() const;

    private:
        struct VplDistKey_ {
            LightPtr key;
            double distance = 0.0;

            VplDistKey_(const LightPtr& key = nullptr, const double distance = 0.0)
                : key(key), distance(distance) {}

            bool operator<(const VplDistKey_& other) const {
                return distance < other.distance;
            }
        };

        struct VplDistKeyLess_ {
            bool operator()(const VplDistKey_& left, const VplDistKey_& right) const {
                return left < right;
            }
        };

        // Used for point light sorting and culling
        using VplDistKeyAllocator_ = StackBasedPoolAllocator<VplDistKey_>;
        typedef std::multiset<VplDistKey_, VplDistKeyLess_, VplDistKeyAllocator_> VplDistMultiSet_;
        typedef std::vector<VplDistKey_, VplDistKeyAllocator_> VplDistVector_;

    private:
        void InitializeVplData_();
        void ClearGBuffer_();
        void InitGBuffer_();
        void UpdateWindowDimensions_();
        void ClearFramebufferData_(const bool);
        void InitPointShadowMaps_();
        // void _InitAllEntityMeshData();
        void InitCoreCSMData_(Pipeline *);
        void InitLights_(Pipeline * s, const VplDistVector_& lights, const size_t maxShadowLights);
        void InitSSAO_();
        void InitAtmosphericShadowing_();
        // void _InitEntityMeshData(RendererEntityData &);
        // void _ClearEntityMeshData();
        void ClearRemovedLightData_();
        void BindShader_(Pipeline *);
        void UnbindShader_();
        void PerformPostFxProcessing_();
        void PerformBloomPostFx_();
        void PerformAtmosphericPostFx_();
        void PerformFxaaPostFx_();
        void PerformTaaPostFx_();
        void PerformGammaTonemapPostFx_();
        void FinalizeFrame_();
        void InitializePostFxBuffers_();
        void RenderBoundingBoxes_(GpuCommandBufferPtr&);
        void RenderBoundingBoxes_(std::unordered_map<RenderFaceCulling, GpuCommandBufferPtr>&);
        void RenderImmediate_(const RenderFaceCulling, GpuCommandBufferPtr&, const CommandBufferSelectionFunction&);
        void Render_(Pipeline&, const RenderFaceCulling, GpuCommandBufferPtr&, const CommandBufferSelectionFunction&, bool isLightInteracting, bool removeViewTranslation = false);
        void Render_(Pipeline&, std::unordered_map<RenderFaceCulling, GpuCommandBufferPtr>&, const CommandBufferSelectionFunction&, bool isLightInteracting, bool removeViewTranslation = false);
        void InitVplFrameData_(const VplDistVector_& perVPLDistToViewer);
        void RenderImmediate_(std::unordered_map<RenderFaceCulling, GpuCommandBufferPtr>&, const CommandBufferSelectionFunction&, const bool reverseCullFace);
        void UpdatePointLights_(
            VplDistMultiSet_&, 
            VplDistVector_&,
            VplDistMultiSet_&,
            VplDistVector_&,
            VplDistMultiSet_&,
            VplDistVector_&,
            std::vector<int, StackBasedPoolAllocator<int>>& visibleVplIndices
        );
        void PerformVirtualPointLightCullingStage1_(VplDistVector_&, std::vector<int, StackBasedPoolAllocator<int>>& visibleVplIndices);
        //void PerformVirtualPointLightCullingStage2_(const std::vector<std::pair<LightPtr, double>>&, const std::vector<int>& visibleVplIndices);
        void PerformVirtualPointLightCullingStage2_(const VplDistVector_&);
        void ComputeVirtualPointLightGlobalIllumination_(const VplDistVector_&, const double);
        void RenderCSMDepth_();
        void RenderQuad_();
        void RenderSkybox_(Pipeline *, const glm::mat4&);
        void RenderSkybox_();
        void RenderForwardPassPbr_();
        void RenderForwardPassFlat_();
        void RenderSsaoOcclude_();
        void RenderSsaoBlur_();
        glm::vec3 CalculateAtmosphericLightPosition_() const;
        void RenderAtmosphericShadowing_();
        ShadowMapCache CreateShadowMap3DCache_(uint32_t resolutionX, uint32_t resolutionY, uint32_t count, bool vpl, const TextureComponentSize&);
        GpuAtlasEntry GetOrAllocateShadowMapForLight_(LightPtr);
        void SetLightShadowMap3D_(LightPtr, GpuAtlasEntry);
        GpuAtlasEntry EvictLightFromShadowMapCache_(LightPtr);
        GpuAtlasEntry EvictOldestLightFromShadowMapCache_(ShadowMapCache&);
        void AddLightToShadowMapCache_(LightPtr);
        void RemoveLightFromShadowMapCache_(LightPtr);
        bool ShadowMapExistsForLight_(LightPtr);
        ShadowMapCache& GetSmapCacheForLight_(LightPtr);
        void RecalculateCascadeData_();
        void ValidateAllShaders_();
    };
}

#endif //STRATUSGFX_RENDERER_H
