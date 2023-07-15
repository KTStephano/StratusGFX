#pragma once

#include "StratusCommon.h"
#include "StratusRendererBackend.h"
#include "StratusEntity.h"
#include "StratusEntityCommon.h"
#include "StratusSystemModule.h"
#include "StratusLight.h"
#include "StratusThread.h"
#include "StratusApplicationThread.h"
#include <cstddef>
#include <memory>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "StratusRenderComponents.h"
#include "StratusGpuCommon.h"
#include "StratusPipeline.h"

namespace stratus {
    struct RendererParams {
        std::string appName;
        Degrees fovy;
        float znear = 1.0f;
        float zfar = 1000.0f;
        bool vsyncEnabled;
    };

    // Public interface of the renderer - manages frame to frame state and manages
    // the backend
    SYSTEM_MODULE_CLASS(RendererFrontend)
    private:
        RendererFrontend(const RendererParams&);

    public:
        void AddLight(const LightPtr&);
        void RemoveLight(const LightPtr&);
        void ClearLights();
        void SetWorldLight(const InfiniteLightPtr&);
        InfiniteLightPtr GetWorldLight();
        void ClearWorldLight();

        void SetCamera(const CameraPtr&);
        CameraPtr GetCamera() const;
        void SetFovY(const Degrees&);
        void SetNearFar(const float znear, const float zfar);
        void SetClearColor(const glm::vec4&);
        RendererSettings GetSettings() const;
        void SetSettings(const RendererSettings&);

        // std::vector<SDL_Event> PollInputEvents();
        // RendererMouseState GetMouseState() const;

        void RecompileShaders();

    private: 
        // SystemModule inteface
        virtual bool Initialize();
        virtual SystemStatus Update(const double);
        virtual void Shutdown();

    private:
        std::unique_lock<std::shared_mutex> LockWrite_() const { return std::unique_lock<std::shared_mutex>(mutex_); }
        std::shared_lock<std::shared_mutex> LockRead_()  const { return std::shared_lock<std::shared_mutex>(mutex_); }
        void AddAllMaterialsForEntity_(const EntityPtr&);
        void RemoveAllMaterialsForEntity_(const EntityPtr&);
        bool AddEntity_(const EntityPtr& p);
        static bool EntityChanged_(const EntityPtr&);
        bool RemoveEntity_(const EntityPtr&);
        void CheckEntitySetForChanges_(std::unordered_set<EntityPtr>&);
        void CopyMaterialToGpuAndMarkForUse_(const MaterialPtr& material, GpuMaterial* gpuMaterial);
        std::unordered_map<RenderFaceCulling, std::vector<GpuDrawElementsIndirectCommand>> GenerateDrawCommands_(RenderComponent *, const size_t, bool&) const;

    private:
        void UpdateViewport_();
        void UpdateCascadeTransforms_();
        void CheckForEntityChanges_();
        void UpdateLights_();
        void UpdateMaterialSet_();
        void MarkStaticLightsDirty_();
        void UpdateDrawCommands_();
        void UpdateVisibility_();
        void UpdateVisibility_(
            Pipeline& pipeline,
            const glm::mat4&, const glm::mat4&, 
            const std::vector<std::unordered_map<RenderFaceCulling, GpuCommandBufferPtr>*>& inDrawCommands,
            const std::vector<std::unordered_map<RenderFaceCulling, GpuCommandBufferPtr>*>& outDrawCommands,
            const std::vector<std::unordered_map<RenderFaceCulling, GpuCommandBufferPtr>*>& selectedLods,
            const std::vector<std::vector<std::unordered_map<RenderFaceCulling, GpuCommandBufferPtr>*>>& drawCommandsPerLod,
            const bool selectLods
            );
        void UpdatePrevFrameModelTransforms_();

    private:
        // These are called by the private entity handler
        friend struct RenderEntityProcess;
        void EntitiesAdded_(const std::unordered_set<stratus::EntityPtr>&);
        void EntitiesRemoved_(const std::unordered_set<stratus::EntityPtr>&);
        void EntityComponentsAdded_(const std::unordered_map<stratus::EntityPtr, std::vector<stratus::EntityComponent *>>&);
        void EntityComponentsEnabledDisabled_(const std::unordered_set<stratus::EntityPtr>&);

    private:
        RendererParams params_;
        std::unordered_set<EntityPtr> entities_;
        // These are entities we need to check for position/orientation/scale updates
        std::unordered_set<EntityPtr> dynamicEntities_;
        //std::vector<GpuMaterial> _gpuMaterials;
        std::unordered_set<LightPtr> lights_;
        std::unordered_set<LightPtr> dynamicLights_;
        std::unordered_set<LightPtr> virtualPointLights_;
        InfiniteLightPtr worldLight_;
        std::unordered_set<LightPtr> lightsToRemove_;
        EntityMeshData flatEntities_;
        EntityMeshData dynamicPbrEntities_;
        EntityMeshData staticPbrEntities_;
        uint64_t lastFrameMaterialIndicesRecomputed_ = 0;
        bool drawCommandsDirty_ = false;
        CameraPtr camera_;
        glm::mat4 projection_ = glm::mat4(1.0f);
        bool viewportDirty_ = true;
        bool recompileShaders_ = false;
        std::shared_ptr<RendererFrame> frame_;
        std::unique_ptr<RendererBackend> renderer_;
        // This forwards entity state changes to the renderer
        EntityProcessHandle entityHandler_;
        // Compute pipeline which performs AABB checks against view frustum
        std::unique_ptr<Pipeline> viscullLodSelect_;
        std::unique_ptr<Pipeline> viscull_;
        std::unique_ptr<Pipeline> updateTransforms_;
        // Used for temporal anti-aliasing
        size_t currentHaltonIndex_ = 0;
        mutable std::shared_mutex mutex_;
    };
}