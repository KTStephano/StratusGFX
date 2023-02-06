#pragma once

#include "StratusCommon.h"
#include "StratusRendererBackend.h"
#include "StratusEntity.h"
#include "StratusEntityCommon.h"
#include "StratusEntity2.h"
#include "StratusRenderNode.h"
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

namespace stratus {
    struct RendererParams {
        std::string appName;
        Degrees fovy;
        float znear = 0.1f;
        float zfar = 750.0f;
        bool vsyncEnabled;
    };

    // Public interface of the renderer - manages frame to frame state and manages
    // the backend
    SYSTEM_MODULE_CLASS(RendererFrontend)
    private:
        RendererFrontend(const RendererParams&);

        struct LightData {
            EntityMeshData visible;
            LightPtr lightCopy;
            bool dirty = true;
        };

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

        void SetGlobalIlluminationEnabled(const bool);
        bool GetGlobalIlluminationEnabled() const;

        // If scatterControl > 1, then backscattered light will be greater than forwardscattered light
        void SetAtmosphericShadowing(float fogDensity, float scatterControl);
        float GetAtmosphericFogDensity() const;
        float GetAtmosphericScatterControl() const;

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
        void _AddAllMaterialsForEntity(const Entity2Ptr&);
        bool _AddEntity(const Entity2Ptr& p);
        static void _AttemptAddEntitiesForLight(const LightPtr& light, LightData& data, const EntityMeshData& entities);
        static bool _EntityChanged(const Entity2Ptr&);
        bool _RemoveEntity(const Entity2Ptr&);
        void _CheckEntitySetForChanges(std::unordered_set<Entity2Ptr>&);
        void _CopyMaterialToGpuAndMarkForUse(const MaterialPtr& material, GpuMaterial* gpuMaterial);
        void _RecalculateMaterialSet();

    private:
        void _UpdateViewport();
        void _UpdateCascadeTransforms();
        void _CheckForEntityChanges();
        void _UpdateLights();
        void _UpdateCameraVisibility();
        void _UpdateCascadeVisibility();
        void _UpdateMaterialSet();

    private:
        // These are called by the private entity handler
        friend struct RenderEntityProcess;
        void _EntitiesAdded(const std::unordered_set<stratus::Entity2Ptr>&);
        void _EntitiesRemoved(const std::unordered_set<stratus::Entity2Ptr>&);
        void _EntityComponentsAdded(const std::unordered_map<stratus::Entity2Ptr, std::vector<stratus::Entity2Component *>>&);
        void _EntityComponentsEnabledDisabled(const std::unordered_set<stratus::Entity2Ptr>&);

    private:
        RendererParams _params;
        std::unordered_set<Entity2Ptr> _entities;
        // These are entities we need to check for position/orientation/scale updates
        std::unordered_set<Entity2Ptr> _dynamicEntities;
        std::unordered_set<MaterialPtr> _dirtyMaterials;
        //std::vector<GpuMaterial> _gpuMaterials;
        std::unordered_map<LightPtr, LightData> _lights;
        std::unordered_set<LightPtr> _virtualPointLights; // data is found in _lights
        InfiniteLightPtr _worldLight;
        std::unordered_set<LightPtr> _lightsToRemove;
        CameraPtr _camera;
        glm::mat4 _projection = glm::mat4(1.0f);
        bool _viewportDirty = true;
        bool _recompileShaders = false;
        std::shared_ptr<RendererFrame> _frame;
        std::unique_ptr<RendererBackend> _renderer;
        // This forwards entity state changes to the renderer
        EntityProcessHandle _entityHandler;
        mutable std::shared_mutex _mutex;
    };
}