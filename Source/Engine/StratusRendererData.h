#pragma once

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
#include "StratusTypes.h"
#include "StratusGraphicsDriver.h"
#include <queue>

namespace stratus {
    enum class RendererCascadeResolution : i32 {
        CASCADE_RESOLUTION_1024  = 1024,
        CASCADE_RESOLUTION_2048  = 2048,
        CASCADE_RESOLUTION_4096  = 4096,
        CASCADE_RESOLUTION_8192  = 8192,
        CASCADE_RESOLUTION_16384 = 16384
    };

    struct RenderMeshContainer {
        RenderComponent * render = nullptr;
        MeshWorldTransforms * transform = nullptr;
        usize meshIndex = 0;
    };

    typedef std::shared_ptr<RenderMeshContainer> RenderMeshContainerPtr;

    // struct RendererEntityData {
    //     std::vector<glm::mat4> modelMatrices;
    //     std::vector<glm::vec3> diffuseColors;
    //     std::vector<glm::vec3> baseReflectivity;
    //     std::vector<f32> roughness;
    //     std::vector<f32> metallic;
    //     GpuArrayBuffer buffers;
    //     usize size = 0;    
    //     // if true, regenerate buffers
    //     bool dirty;
    // };

    struct RendererMouseState {
        i32 x;
        i32 y;
        u32 mask;
    };

    typedef std::unordered_map<EntityPtr, std::vector<RenderMeshContainerPtr>> EntityMeshData;

    struct RendererCascadeData {
        GpuCommandReceiveManagerPtr drawCommandsFrustumCulled;
        GpuCommandReceiveManagerPtr drawCommandsFinal;
        // Use during shadow map rendering
        glm::mat4 projectionViewRender;
        glm::mat4 invProjectionViewRender;
        // Use during shadow map sampling
        glm::mat4 projectionViewSample;
        // Transforms from a cascade 0 sampling coordinate to the current cascade
        glm::mat4 sampleCascade0ToCurrent;
        glm::vec4 cascadePlane;
        glm::vec3 cascadePositionLightSpace;
        glm::vec3 cascadePositionCameraSpace;
        f32 cascadeDiameter;
        f32 cascadeBegins;
        f32 cascadeEnds;
        f32 cascadeZDifference;
    };

    static inline u32 ComputeFlatVirtualIndex(const u32 x, const u32 y, const u32 maxX) {
        return x + y * maxX;
    }

    struct VirtualIndex2DUpdateQueue {
        VirtualIndex2DUpdateQueue(const u32 maxX = 1, const u32 maxY = 1)
            : maxX_(maxX), maxY_(maxY) {}

        void PushFront(const u32 x, const u32 y, const u32 count = 0) {
            const u32 index = ComputeFlatVirtualIndex(x, y, maxX_);
            if (existing_.find(index) != existing_.end()) return;

            indexQueue_.push_front(std::make_pair(std::make_pair(x, y), count));
            auto it = indexQueue_.begin();
            existing_.insert(std::make_pair(index, it));
        }

        void PushBack(const u32 x, const u32 y, const u32 count = 0) {
            const u32 index = ComputeFlatVirtualIndex(x, y, maxX_);
            if (existing_.find(index) != existing_.end()) return;

            indexQueue_.push_back(std::make_pair(std::make_pair(x, y), count));
            auto it = indexQueue_.end();
            --it;
            existing_.insert(std::make_pair(index, it));
        }

        std::pair<std::pair<u32, u32>, u32> PopFront() {
            if (Size() == 0) return std::make_pair(std::make_pair(u32(0), u32(0)), 0);

            auto front = Front();
            indexQueue_.pop_front();
            existing_.erase(ComputeFlatVirtualIndex(front.first.first, front.first.second, maxX_));

            return front;
        }

        std::pair<std::pair<u32, u32>, u32> Front() const {
            if (Size() == 0) return std::make_pair(std::make_pair(u32(0), u32(0)), u32(0));
            return indexQueue_.front();
        }

        void Erase(const u32 x, const u32 y) {
            const u32 index = x + y * maxX_;
            auto existing = existing_.find(index);
            if (existing == existing_.end()) return;

            auto it = existing->second;
            indexQueue_.erase(it);
            existing_.erase(index);
        }

        // Removes any element not contained in both this set and the other
        template<typename Set>
        void SetIntersection(const Set& other) {
            for (auto it = indexQueue_.begin(); it != indexQueue_.end();) {
                const auto index = ComputeFlatVirtualIndex(it->first.first, it->first.second, maxX_);
                auto old = it;
                ++it;
                if (other.find(index) == other.end()) {
                    Erase(old->first.first, old->first.second);
                }
            }
        }

        void Clear() {
            indexQueue_.clear();
            existing_.clear();
        }

        usize Size() const {
            return indexQueue_.size();
        }

    private:
        std::list<std::pair<std::pair<u32, u32>, u32>> indexQueue_;
        std::unordered_map<u32, std::list<std::pair<std::pair<u32, u32>, u32>>::iterator> existing_;
        u32 maxX_;
        u32 maxY_;
    };

    // Manages the data for the GI probes
    class RendererProbeManager {

    };

    //struct RendererCascadeContainer {
    //    FrameBuffer fbo;
    //    Texture vsm;
    //    GpuBuffer prevFramePageResidencyTable;
    //    GpuBuffer currFramePageResidencyTable;
    //    UnsafePtr<VirtualIndex2DUpdateQueue> pageGroupUpdateQueue;
    //    UnsafePtr<VirtualIndex2DUpdateQueue> backPageGroupUpdateQueue;
    //    // Texture is split into pages which are combined
    //    // into page groups for geometry culling purposes
    //    u32 numPageGroupsX = 32;
    //    u32 numPageGroupsY = 32;
    //    std::vector<glm::mat4> tiledProjectionMatrices;
    //    GpuBuffer numDrawCalls;
    //    GpuBuffer numPagesToCommit;
    //    GpuBuffer pagesToCommitList;
    //    GpuBuffer pageGroupsToRender;
    //    GpuBuffer pageBoundingBox;
    //    std::vector<RendererCascadeData> cascades;
    //    glm::vec4 cascadeShadowOffsets[2];
    //    u32 cascadeResolutionXY;
    //    InfiniteLightPtr worldLight;
    //    CameraPtr worldLightCamera;
    //    glm::vec3 worldLightDirectionCameraSpace;
    //    f32 znear;
    //    f32 zfar;
    //    bool regenerateFbo;    
    //};

    struct RendererVsmCascadeData {
        glm::mat4 projectionViewRender;
        glm::mat4 invProjectionViewRender;
        glm::mat4 projection;
    };

    struct RendererVsmContainer {
        // Contains one list of commands per clip cascade in a single buffer
        //GpuCommandReceiveManagerPtr drawCommandsFrustumCulled;
        GpuCommandReceiveManagerPtr drawCommandsFinal;
        // Use during shadow map rendering
        std::vector<RendererVsmCascadeData> cascades;
        // Use during shadow map sampling
        glm::mat4 projectionViewSample;
        glm::mat4 viewTransform;
        glm::vec3 cascadePositionLightSpace;
        glm::vec3 cascadePositionCameraSpace;
        // Should be a power of 2
        u32 updateDivisor = 2; // 1 = update everything every frame
        u32 currUpdateX = 0;
        u32 currUpdateY = 0;
        f32 baseCascadeDiameter;
        FrameBuffer fbo;
        Texture vsm;
        // Hierarchical page buffer (for culling)
        Texture hpb;
        GpuHostFence prevFrameFence;
        // GpuBuffer prevFramePageResidencyTable;
        GpuBuffer pageResidencyTable;
        std::vector<UnsafePtr<VirtualIndex2DUpdateQueue>> pageGroupUpdateQueue;
        std::vector<UnsafePtr<VirtualIndex2DUpdateQueue>> backPageGroupUpdateQueue;
        // Texture is split into pages which are combined
        // into page groups for geometry culling purposes
        u32 numPageGroupsX = 32;
        u32 numPageGroupsY = 32;
        std::vector<glm::mat4> tiledProjectionMatrices;
        glm::vec3 lightSpacePrevPosition = glm::vec3(0.0f);
        glm::vec2 ndcClipOriginDifference = glm::vec2(0.0f);
        // GpuBuffer numDrawCalls;
        GpuBuffer numPagesToCommit;
        GpuBuffer pagesToCommitList;
        GpuBuffer numPagesFree;
        GpuBuffer pagesFreeList;
        GpuBuffer pageGroupsToRender;
        GpuBuffer pageBoundingBox;
        glm::vec4 cascadeShadowOffsets[2];
        u32 cascadeResolutionXY;
        InfiniteLightPtr worldLight;
        CameraPtr worldLightCamera;
        glm::vec3 worldLightDirectionCameraSpace;
        f32 znear;
        f32 zfar;
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
            auto existing = existing_.find(ptr);
            if (existing == existing_.end()) return;
            auto it = existing->second;
            queue_.erase(it);
            existing_.erase(ptr);
            //for (auto it = queue_.begin(); it != queue_.end(); ++it) {
            //    const LightPtr& light = *it;
            //    if (ptr == light) {
            //        queue_.erase(it);
            //        return;
            //    }
            //}
        }

        // In case all lights need to be removed without being updated
        void Clear() {
            queue_.clear();
            existing_.clear();
        }

        usize Size() const {
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
        usize perFrameMaxScratchMemoryBytes = 134217728; // 128 mb

        f32 GetEmissionStrength() const {
            return emissionStrength_;
        }

        void SetEmissionStrength(const f32 strength) {
            emissionStrength_ = std::max<f32>(strength, 0.0f);
        }

        f32 GetEmissiveTextureMultiplier() const {
            return emissiveTextureMultiplier_;
        }

        void SetEmissiveTextureMultiplier(const f32 multiplier) {
            emissiveTextureMultiplier_ = std::max<f32>(multiplier, 0.0f);
        }

        glm::vec3 GetFogColor() const {
            return fogColor_;
        }

        f32 GetFogDensity() const {
            return fogDensity_;
        }

        void SetFogColor(const glm::vec3& color) {
            fogColor_ = glm::vec3(
                std::max<f32>(color[0], 0.0f),
                std::max<f32>(color[1], 0.0f),
                std::max<f32>(color[2], 0.0f)
            );
        }

        void SetFogDensity(const f32 density) {
            fogDensity_ = std::max<f32>(density, 0.0f);
        }

        glm::vec3 GetSkyboxColorMask() const {
            return skyboxColorMask_;
        }

        f32 GetSkyboxIntensity() const {
            return skyboxIntensity_;
        }

        void SetSkyboxColorMask(const glm::vec3& mask) {
            skyboxColorMask_ = glm::vec3(
                std::max<f32>(mask[0], 0.0f),
                std::max<f32>(mask[1], 0.0f),
                std::max<f32>(mask[2], 0.0f)
            );
        }

        void SetSkyboxIntensity(const f32 intensity) {
            skyboxIntensity_ = std::max<f32>(intensity, 0.0f);
        }

        f32 GetMinRoughness() const {
            return minRoughness_;
        }

        void SetMinRoughness(const f32 roughness) {
            minRoughness_ = std::max<f32>(roughness, 0.0f);
        }

        void SetAlphaDepthTestThreshold(const f32 threshold) {
            alphaDepthTestThreshold_ = std::clamp<f32>(threshold, 0.0f, 1.0f);
        }

        f32 GetAlphaDepthTestThreshold() const {
            return alphaDepthTestThreshold_;
        }

        // Values of 1.0 mean that GI occlusion will result in harsh shadow cutoffs
        // Values < 1.0 effectively brighten the scene
        void SetMinGiOcclusionFactor(const f32 value) {
            minGiOcclusionFactor_ = std::clamp<f32>(value, 0.0f, 1.0f);
        }

        f32 GetMinGiOcclusionFactor() const {
            return minGiOcclusionFactor_;
        }

    private:
        // These are all values we need to range check when they are set
        glm::vec3 fogColor_ = glm::vec3(0.5f);
        f32 fogDensity_ = 0.0f;
        f32 emissionStrength_ = 0.0f;
        glm::vec3 skyboxColorMask_ = glm::vec3(1.0f);
        f32 skyboxIntensity_ = 3.0f;
        f32 minRoughness_ = 0.08f;
        f32 alphaDepthTestThreshold_ = 0.5f;
        // This works as a multiplicative effect on top of emission strength
        f32 emissiveTextureMultiplier_ = 1.0f;
        f32 minGiOcclusionFactor_ = 0.95f;
    };

    // Represents data for current active frame
    struct RendererFrame {
        u32 viewportWidth;
        u32 viewportHeight;
        Radians fovy;
        CameraPtr camera;
        std::vector<glm::vec4, StackBasedPoolAllocator<glm::vec4>> viewFrustumPlanes;
        GpuMaterialBufferPtr materialInfo;
        //RendererCascadeContainer csc;
        RendererVsmContainer vsmc;
        GpuCommandManagerPtr drawCommands;
        SpatialLightMap lights = SpatialLightMap(256);
        //std::unordered_set<LightPtr> lights;
        //std::unordered_set<LightPtr> virtualPointLights; // data is in lights
        LightUpdateQueue lightsToUpdate; // shadow map data is invalid
        std::unordered_set<LightPtr> lightsToRemove;
        f32 znear;
        f32 zfar;
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
}