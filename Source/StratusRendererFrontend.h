#pragma once

#include "StratusCommon.h"
#include "StratusRendererBackend.h"
#include "StratusEntity.h"
#include "StratusRenderNode.h"
#include "StratusSystemModule.h"
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
    class RendererFrontend : public SystemModule {
        friend class Engine;
        RendererFrontend(const RendererParams&);

        struct LightData {
            std::unordered_set<EntityView> visible;
            LightPtr lightCopy;
            bool dirty = true;
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
        void SetWorldLight(const InfiniteLightPtr&);
        void ClearWorldLight();

        void SetCamera(const CameraPtr&);
        void SetViewportDims(const uint32_t width, const uint32_t height);
        void SetFovY(const Degrees&);
        void SetNearFar(const float znear, const float zfar);
        void SetVsyncEnabled(const bool);
        void SetClearColor(const glm::vec4&);

        // If scatterControl > 1, then backscattered light will be greater than forwardscattered light
        void SetAtmosphericShadowing(float fogDensity, float scatterControl);
        float GetAtmosphericFogDensity() const;
        float GetAtmosphericScatterControl() const;

        std::vector<SDL_Event> PollInputEvents();
        RendererMouseState GetMouseState() const;

        // SystemModule inteface
        virtual const char * Name() const {
            return "Renderer";
        }

        virtual bool Initialize();
        virtual SystemStatus Update(const double);
        virtual void Shutdown();

        void RecompileShaders();

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
        void _SwapFrames();

    private:
        static RendererFrontend * _instance;

        RendererParams _params;
        std::unordered_map<EntityView, EntityStateData> _staticPbrEntities;
        std::unordered_map<EntityView, EntityStateData> _dynamicPbrEntities;
        std::unordered_map<EntityView, EntityStateData> _flatEntities;
        std::unordered_map<LightPtr, LightData> _lights;
        InfiniteLightPtr _worldLight;
        std::unordered_set<LightPtr> _lightsToRemove;
        CameraPtr _camera;
        glm::mat4 _projection = glm::mat4(1.0f);
        bool _staticPbrDirty = true;
        bool _dynamicPbrDirty = true;
        bool _lightsDirty = true;
        bool _viewportDirty = true;
        bool _recompileShaders = false;
        std::shared_ptr<RendererFrame> _frame;
        std::shared_ptr<RendererFrame> _prevFrame;
        std::unique_ptr<RendererBackend> _renderer;
        std::vector<SDL_Event> _events;
        RendererMouseState _mouse;
        mutable std::shared_mutex _mutex;
    };
}