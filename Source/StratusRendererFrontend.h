#pragma once

#include "StratusCommon.h"
#include "StratusRendererBackend.h"
#include "StratusEntity.h"
#include "StratusRenderNode.h"
#include "StratusLight.h"
#include "StratusThread.h"
#include <cstddef>
#include <memory>
#include <shared_mutex>

namespace stratus {
    struct RendererParams {
        uint32_t viewportWidth;
        uint32_t viewportHeight;
        std::string appName;
        Degrees fovy;
        float znear = 0.1f;
        float zfar = 1000.0f;
        bool vsyncEnabled;
    };

    // Public interface of the renderer - manages frame to frame state and manages
    // the backend
    class RendererFrontend {
        friend class Engine;
        RendererFrontend(const RendererParams&);

        struct LightData {
            std::unordered_set<EntityView> visible;
            LightPtr lightCopy;
            bool dirty = true;
        };

        struct WorldLightData {
            bool enabled = false;
            glm::vec3 color = glm::vec3(1.0f);
            float intensity = 1.0f;
            Rotation rotation;
        };

        struct EntityStateData {
            glm::vec3 lastPosition;
            glm::vec3 lastScale;
            glm::vec3 lastRotation;
        };

    public:
        static RendererFrontend * Instance() { return _instance; }

        void AddStaticEntity(const EntityPtr&);
        void AddDynamicEntity(const EntityPtr&);
        void RemoveEntity(const EntityPtr&);
        void ClearEntities();

        void AddLight(const LightPtr&);
        void RemoveLight(const LightPtr&);
        void ClearLights();
        void SetWorldLightingEnabled(const bool);
        void SetWorldLightColor(const glm::vec3&);
        void SetWorldLightIntensity(float);
        void SetWorldLightRotation(const Rotation&);

        void SetCamera(const CameraPtr&);
        void SetViewportDims(const uint32_t width, const uint32_t height);
        void SetFovY(const Degrees&);
        void SetNearFar(const float znear, const float zfar);
        void SetVsyncEnabled(const bool);
        void SetClearColor(const glm::vec4&);

        std::vector<SDL_Event> PollInputEvents();
        RendererMouseState GetMouseState() const;

        void Update(const double);

        void QueueRendererThreadTask(const Thread::ThreadFunction&);

    private:
        std::unique_lock<std::shared_mutex> _LockWrite() const { return std::unique_lock<std::shared_mutex>(_mutex); }
        std::shared_lock<std::shared_mutex> _LockRead()  const { return std::shared_lock<std::shared_mutex>(_mutex); }
        static void _AddEntity(const EntityPtr& p, bool& pbrDirty, std::unordered_map<EntityView, EntityStateData>& pbr, std::unordered_map<EntityView, EntityStateData>& flat, std::unordered_map<LightPtr, LightData>& lights);
        static void _AttemptAddEntitiesForLight(const LightPtr& light, LightData& data, const std::unordered_map<EntityView, EntityStateData>& entities);
        static bool _EntityChanged(const EntityView&, const EntityStateData&);
        void _CheckEntitySetForChanges(std::unordered_map<EntityView, EntityStateData>&, bool&);

    private:
        void _UpdateViewport();
        void _UpdateCascadeTransforms();
        void _CheckForEntityChanges();
        void _UpdateLights();
        void _UpdateCameraVisibility();
        void _UpdateCascadeVisibility();

    private:
        static RendererFrontend * _instance;

        RendererParams _params;
        std::unordered_map<EntityView, EntityStateData> _staticPbrEntities;
        std::unordered_map<EntityView, EntityStateData> _dynamicPbrEntities;
        std::unordered_map<EntityView, EntityStateData> _flatEntities;
        std::unordered_map<LightPtr, LightData> _lights;
        std::unordered_set<LightPtr> _lightsToRemove;
        CameraPtr _camera;
        WorldLightData _worldLight;
        glm::mat4 _projection = glm::mat4(1.0f);
        bool _staticPbrDirty = true;
        bool _dynamicPbrDirty = true;
        bool _lightsDirty = true;
        bool _viewportDirty = true;
        std::shared_ptr<RendererFrame> _frame;
        std::unique_ptr<RendererBackend> _renderer;
        std::vector<SDL_Event> _events;
        RendererMouseState _mouse;
        std::vector<Thread::ThreadFunction> _rendererTasks;
        mutable std::shared_mutex _mutex;
    };
}