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
        void SetVsyncEnabled(const bool);
        void SetClearColor(const glm::vec4&);
        void SetSkybox(const TextureHandle&);
        void SetSkyboxColorMask(const glm::vec3&);
        void SetSkyboxIntensity(const float);
        void SetFogColor(const glm::vec3&);
        void SetFogDensity(const float);

        void SetGlobalIlluminationEnabled(const bool);
        bool GetGlobalIlluminationEnabled() const;

        // std::vector<SDL_Event> PollInputEvents();
        // RendererMouseState GetMouseState() const;

        void RecompileShaders();

    private:
        // SystemModule inteface
        virtual bool Initialize();
        virtual SystemStatus Update(const double);
        virtual void Shutdown();

    private:
        std::unique_lock<std::shared_mutex> _LockWrite() const { return std::unique_lock<std::shared_mutex>(_mutex); }
        std::shared_lock<std::shared_mutex> _LockRead()  const { return std::shared_lock<std::shared_mutex>(_mutex); }
        void _AddAllMaterialsForEntity(const EntityPtr&);
        bool _AddEntity(const EntityPtr& p);
        static bool _EntityChanged(const EntityPtr&);
        bool _RemoveEntity(const EntityPtr&);
        void _CheckEntitySetForChanges(std::unordered_set<EntityPtr>&);
        void _CopyMaterialToGpuAndMarkForUse(const MaterialPtr& material, GpuMaterial* gpuMaterial);
        void _RecalculateMaterialSet();
        std::unordered_map<RenderFaceCulling, std::vector<GpuDrawElementsIndirectCommand>> _GenerateDrawCommands(RenderComponent *, const size_t, bool&) const;

    private:
        void _UpdateViewport();
        void _UpdateCascadeTransforms();
        void _CheckForEntityChanges();
        void _UpdateLights();
        void _UpdateMaterialSet();
        void _MarkStaticLightsDirty();
        void _UpdateDrawCommands();
        void _UpdateVisibility();
        void _UpdateVisibility(
            Pipeline& pipeline,
            const glm::mat4&, const glm::mat4&, 
            const std::vector<std::unordered_map<RenderFaceCulling, GpuCommandBufferPtr>*>& drawCommands,
            const std::vector<std::vector<std::unordered_map<RenderFaceCulling, GpuCommandBufferPtr>*>>& drawCommandsPerLod
            );

    private:
        // These are called by the private entity handler
        friend struct RenderEntityProcess;
        void _EntitiesAdded(const std::unordered_set<stratus::EntityPtr>&);
        void _EntitiesRemoved(const std::unordered_set<stratus::EntityPtr>&);
        void _EntityComponentsAdded(const std::unordered_map<stratus::EntityPtr, std::vector<stratus::EntityComponent *>>&);
        void _EntityComponentsEnabledDisabled(const std::unordered_set<stratus::EntityPtr>&);

    private:
        RendererParams _params;
        std::unordered_set<EntityPtr> _entities;
        // These are entities we need to check for position/orientation/scale updates
        std::unordered_set<EntityPtr> _dynamicEntities;
        //std::vector<GpuMaterial> _gpuMaterials;
        std::unordered_set<LightPtr> _lights;
        std::unordered_set<LightPtr> _dynamicLights;
        std::unordered_set<LightPtr> _virtualPointLights;
        InfiniteLightPtr _worldLight;
        std::unordered_set<LightPtr> _lightsToRemove;
        EntityMeshData _flatEntities;
        EntityMeshData _dynamicPbrEntities;
        EntityMeshData _staticPbrEntities;
        uint64_t _lastFrameMaterialIndicesRecomputed = 0;
        bool _materialsDirty = false;
        bool _drawCommandsDirty = false;
        CameraPtr _camera;
        glm::mat4 _projection = glm::mat4(1.0f);
        bool _viewportDirty = true;
        bool _recompileShaders = false;
        std::shared_ptr<RendererFrame> _frame;
        std::unique_ptr<RendererBackend> _renderer;
        // This forwards entity state changes to the renderer
        EntityProcessHandle _entityHandler;
        // Compute pipeline which performs AABB checks against view frustum
        std::unique_ptr<Pipeline> _viscullLodSelect;
        std::unique_ptr<Pipeline> _viscull;
        mutable std::shared_mutex _mutex;
    };
}