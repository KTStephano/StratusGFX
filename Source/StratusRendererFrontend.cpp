#include "StratusRendererFrontend.h"
#include "StratusUtils.h"
#include "StratusLog.h"
#include "StratusWindow.h"
#include "StratusTransformComponent.h"
#include "StratusRenderComponents.h"
#include "StratusResourceManager.h"
#include "StratusEngine.h"
#include "StratusEntity.h"
#include "StratusEntityProcess.h"
#include "StratusEntityManager.h"
#include "StratusGraphicsDriver.h"

#include <algorithm>

namespace stratus {
    struct RenderEntityProcess : public EntityProcess {
        virtual ~RenderEntityProcess() = default;

        virtual void Process(const double deltaSeconds) {}

        void EntitiesAdded(const std::unordered_set<stratus::EntityPtr>& e) override {
            auto rf = INSTANCE(RendererFrontend);
            if (rf) rf->_EntitiesAdded(e);
        }

        void EntitiesRemoved(const std::unordered_set<stratus::EntityPtr>& e) override {
            auto rf = INSTANCE(RendererFrontend);
            if (rf) rf->_EntitiesRemoved(e);
        }

        void EntityComponentsAdded(const std::unordered_map<stratus::EntityPtr, std::vector<stratus::EntityComponent *>>& e) override {
            auto rf = INSTANCE(RendererFrontend);
            if (rf) rf->_EntityComponentsAdded(e);
        }

        void EntityComponentsEnabledDisabled(const std::unordered_set<stratus::EntityPtr>& e) override {
            auto rf = INSTANCE(RendererFrontend);
            if (rf) rf->_EntityComponentsEnabledDisabled(e);
        }
    };

    static void InitializeMeshTransformComponent(const EntityPtr& p) {
        if (!p->Components().ContainsComponent<MeshWorldTransforms>()) p->Components().AttachComponent<MeshWorldTransforms>();

        auto global = p->Components().GetComponent<GlobalTransformComponent>().component;
        auto rc = p->Components().GetComponent<RenderComponent>().component;
        auto meshTransform = p->Components().GetComponent<MeshWorldTransforms>().component;
        meshTransform->transforms.resize(rc->GetMeshCount());

        for (size_t i = 0; i < rc->GetMeshCount(); ++i) {
            meshTransform->transforms[i] = global->GetGlobalTransform() * rc->meshes->transforms[i];
        }
    }

    static bool IsStaticEntity(const EntityPtr& p) {
        auto sc = p->Components().GetComponent<StaticObjectComponent>();
        return sc.component != nullptr && sc.status == EntityComponentStatus::COMPONENT_ENABLED;
    }

    static glm::vec3 GetWorldTransform(const EntityPtr& p, const size_t meshIndex) {
        return glm::vec3(GetTranslate(p->Components().GetComponent<MeshWorldTransforms>().component->transforms[meshIndex]));
    }

    static MeshPtr GetMesh(const EntityPtr& p, const size_t meshIndex) {
        return p->Components().GetComponent<RenderComponent>().component->GetMesh(meshIndex);
    }

    static bool InsertMesh(EntityMeshData& map, const EntityPtr& p, const size_t meshIndex) {
        auto it = map.find(p);
        if (it == map.end()) {
            map.insert(std::make_pair(p, std::vector<RenderMeshContainerPtr>()));
            it = map.find(p);
        }

        auto rc = p->Components().GetComponent<RenderComponent>().component;
        auto mt = p->Components().GetComponent<MeshWorldTransforms>().component;

        // Check if it is already present
        for (auto& mc : it->second) {
            if (mc->meshIndex == meshIndex) return false;
        }

        RenderMeshContainerPtr c(new RenderMeshContainer());
        c->render = rc;
        c->transform = mt;
        c->meshIndex = meshIndex;
        it->second.push_back(std::move(c));

        return true;
    }

    static bool RemoveMesh(EntityMeshData& map, const EntityPtr& p, const size_t meshIndex) {
        auto it = map.find(p);
        if (it == map.end()) return false;

        for (auto mit = it->second.begin(); mit != it->second.end(); ++mit) {
            if ((*mit)->meshIndex == meshIndex) {
                it->second.erase(mit);
                return true;
            }
        }

        return false;
    }

    RendererFrontend::RendererFrontend(const RendererParams& p)
        : _params(p) {
    }

    void RendererFrontend::_AddAllMaterialsForEntity(const EntityPtr& p) {
        _materialsDirty = true;
        RenderComponent * c = p->Components().GetComponent<RenderComponent>().component;
        for (size_t i = 0; i < c->GetMaterialCount(); ++i) {
            _frame->materialInfo.availableMaterials.insert(c->GetMaterialAt(i));
        }
    }

    void RendererFrontend::_EntitiesAdded(const std::unordered_set<stratus::EntityPtr>& e) {
        auto ul = _LockWrite();
        bool added = false;
        for (auto ptr : e) {
            added |= _AddEntity(ptr);
        }

        _drawCommandsDirty = _drawCommandsDirty || added;
    }

    void RendererFrontend::_EntitiesRemoved(const std::unordered_set<stratus::EntityPtr>& e) {
        auto ul = _LockWrite();
        bool removed = false;
        for (auto& ptr : e) {
            removed = removed || _RemoveEntity(ptr);
        }

        _drawCommandsDirty = _drawCommandsDirty || removed;
        if (removed) {
            _RecalculateMaterialSet();
        }
    }

    void RendererFrontend::_EntityComponentsAdded(const std::unordered_map<stratus::EntityPtr, std::vector<stratus::EntityComponent *>>& e) {
        auto ul = _LockWrite();
        bool changed = false;
        for (auto& entry : e) {
            auto ptr = entry.first;
            if (_RemoveEntity(ptr)) {
                changed = true;
                _AddEntity(ptr);
            }
        }

        _drawCommandsDirty = _drawCommandsDirty || changed;
        if (changed) {
            _RecalculateMaterialSet();
        }
    }

    void RendererFrontend::_EntityComponentsEnabledDisabled(const std::unordered_set<stratus::EntityPtr>& e) {
        auto ul = _LockWrite();
        bool changed = false;
        for (auto& ptr : e) {
            if (_RemoveEntity(ptr)) {
                changed = true;
                _AddEntity(ptr);
            }
        }

        _drawCommandsDirty = _drawCommandsDirty || changed;
        if (changed) {
            _RecalculateMaterialSet();
        }
    }

    bool RendererFrontend::_AddEntity(const EntityPtr& p) {
        if (p == nullptr || _entities.find(p) != _entities.end()) return false;
        
        if (IsRenderable(p)) {
            InitializeMeshTransformComponent(p);

            _entities.insert(p);

            const bool isStatic = IsStaticEntity(p);

            if (!isStatic) {
                _dynamicEntities.insert(p);
            }

            _AddAllMaterialsForEntity(p);
            //_renderComponents.insert(p->Components().GetComponent<RenderComponent>().component);
            
            if (IsLightInteracting(p)) {
                for (size_t i = 0; i < GetMeshCount(p); ++i) {
                    if (isStatic) InsertMesh(_staticPbrEntities, p, i);
                    else InsertMesh(_dynamicPbrEntities, p, i);

                    for (auto& entry : _lights) {
                        if (!entry->castsShadows()) continue;
                        auto pos = entry->GetPosition();
                        if ((isStatic && entry->IsStaticLight()) || !entry->IsStaticLight()) {
                            if (glm::distance(GetWorldTransform(p, i), pos) < entry->getRadius()) {
                                _frame->lightsToUpate.PushBack(entry);
                            }
                        }
                    }
                }
            }
            else {
                for (size_t i = 0; i < GetMeshCount(p); ++i) {
                    InsertMesh(_flatEntities, p, i);
                }
            }

            return true;
        }
        else {
            return false;
        }
    }

    bool RendererFrontend::_RemoveEntity(const EntityPtr& p) {
        if (p == nullptr || _entities.find(p) == _entities.end()) return false;

        _entities.erase(p);
        _dynamicEntities.erase(p);
        _dynamicPbrEntities.erase(p);
        _staticPbrEntities.erase(p);
        _flatEntities.erase(p);

        for (auto& entry : _lights) {
            if (!entry->castsShadows()) continue;
            //if (entry.second.visible.erase(p)) {
                if (entry->IsStaticLight()) {
                    if (IsStaticEntity(p)) {
                        _frame->lightsToUpate.PushBack(entry);
                    }
                }
                else {
                    _frame->lightsToUpate.PushBack(entry);
                }
            //}
        }

        return true;
    }

    void RendererFrontend::AddLight(const LightPtr& light) {
        auto ul = _LockWrite();
        if (_lights.find(light) != _lights.end()) return;

        _lights.insert(light);
        _frame->lights.insert(light);

        if ( light->IsVirtualLight() ) _virtualPointLights.insert(light);

        if ( !light->IsStaticLight() ) _dynamicLights.insert(light);

        if ( !light->castsShadows() ) return;

        _frame->lightsToUpate.PushBack(light);

        //_AttemptAddEntitiesForLight(light, data, _frame->instancedPbrMeshes);
    }

    void RendererFrontend::RemoveLight(const LightPtr& light) {
        auto ul = _LockWrite();
        if (_lights.find(light) == _lights.end()) return;
        _lights.erase(light);
        _dynamicLights.erase(light);
        _virtualPointLights.erase(light);
        _lightsToRemove.insert(light);
        _frame->lightsToUpate.Erase(light);
    }

    void RendererFrontend::ClearLights() {
        auto ul = _LockWrite();
        for (auto& light : _lights) {
            _lightsToRemove.insert(light);
        }
        _lights.clear();
        _dynamicLights.clear();
        _virtualPointLights.clear();
        _frame->lightsToUpate.Clear();
    }

    void RendererFrontend::SetWorldLight(const InfiniteLightPtr& light) {
        if (light == nullptr) return;
        auto ul = _LockWrite();
        _worldLight = light;
    }

    InfiniteLightPtr RendererFrontend::GetWorldLight() {
        auto sl = _LockRead();
        return _worldLight;
    }

    void RendererFrontend::ClearWorldLight() {
        auto ul = _LockWrite();
        // Create a dummy world light that is disabled
        _worldLight = InfiniteLightPtr(new InfiniteLight(false));
    }

    void RendererFrontend::SetCamera(const CameraPtr& camera) {
        auto ul = _LockWrite();
        _camera = camera;
    }

    CameraPtr RendererFrontend::GetCamera() const {
        auto sl = _LockRead();
        return _camera;
    }

    void RendererFrontend::SetFovY(const Degrees& fovy) {
        auto ul = _LockWrite();
        _params.fovy = fovy;
        _viewportDirty = true;
    }

    void RendererFrontend::SetNearFar(const float znear, const float zfar) {
        auto ul = _LockWrite();
        _params.znear = znear;
        _params.zfar  = zfar;
        _viewportDirty = true;
    }

    void RendererFrontend::SetVsyncEnabled(const bool enabled) {
        auto ul = _LockWrite();
        _params.vsyncEnabled = enabled;
        _frame->vsyncEnabled = enabled;
    }

    void RendererFrontend::SetClearColor(const glm::vec4& color) {
        auto ul = _LockWrite();
        _frame->clearColor = color;
    }

    void RendererFrontend::SetSkybox(const TextureHandle& skybox) {
        auto ul = _LockWrite();
        _frame->skybox = skybox;
    }

    void RendererFrontend::SetAtmosphericShadowing(float fogDensity, float scatterControl) {
        auto ul = _LockWrite();
        _frame->atmospheric.fogDensity = std::max(0.0f, std::min(fogDensity, 1.0f));
        _frame->atmospheric.scatterControl = scatterControl;
    }

    float RendererFrontend::GetAtmosphericFogDensity() const {
        auto sl = _LockRead();
        return _frame->atmospheric.fogDensity;
    }

    float RendererFrontend::GetAtmosphericScatterControl() const {
        auto sl = _LockRead();
        return _frame->atmospheric.scatterControl;
    }

    void RendererFrontend::SetGlobalIlluminationEnabled(const bool enabled) {
        auto ul = _LockWrite();
        _frame->globalIlluminationEnabled = enabled;
    }

    bool RendererFrontend::GetGlobalIlluminationEnabled() const {
        auto sl = _LockRead();
        return _frame->globalIlluminationEnabled;
    }

    SystemStatus RendererFrontend::Update(const double deltaSeconds) {
        CHECK_IS_APPLICATION_THREAD();

        auto ul = _LockWrite();
        if (_camera == nullptr) return SystemStatus::SYSTEM_CONTINUE;

        _camera->update(deltaSeconds);
        _frame->camera = _camera->Copy();

        _UpdateViewport();
        _UpdateCascadeTransforms();
        _CheckForEntityChanges();
        _UpdateLights();
        _UpdateMaterialSet();
        _UpdateDrawCommands();

        //_SwapFrames();

        // Check for shader recompile request
        if (_recompileShaders) {
            _renderer->RecompileShaders();
            _recompileShaders = false;
        }

        // Begin the new frame
        _renderer->Begin(_frame, true);

        // Complete the frame
        _renderer->RenderScene();
        _renderer->End();

        // This needs to be unset
        _frame->csc.regenerateFbo = false;

        return SystemStatus::SYSTEM_CONTINUE;
    }

    bool RendererFrontend::Initialize() {
        CHECK_IS_APPLICATION_THREAD();
        // Create the renderer on the renderer thread only
        _renderer = std::make_unique<RendererBackend>(Window::Instance()->GetWindowDims().first, Window::Instance()->GetWindowDims().second, _params.appName);

        _frame = std::make_shared<RendererFrame>();

        // 4 cascades total
        _frame->csc.cascades.resize(4);
        _frame->csc.cascadeResolutionXY = 2048;
        _frame->csc.regenerateFbo = true;

        // Set materials per frame and initialize material buffer
        _frame->materialInfo.maxMaterials = 4096;
        const Bitfield flags = GPU_DYNAMIC_DATA | GPU_MAP_READ | GPU_MAP_WRITE;
        _frame->materialInfo.materialsBuffer = GpuBuffer(nullptr, sizeof(GpuMaterial) * _frame->materialInfo.maxMaterials, flags);

        // Set up draw command buffers
        std::vector<RenderFaceCulling> culling{
            RenderFaceCulling::CULLING_CCW,
            RenderFaceCulling::CULLING_CW,
            RenderFaceCulling::CULLING_NONE  
        };

        for (auto cull : culling) {
            _frame->instancedFlatMeshes.insert(std::make_pair(cull, GpuCommandBufferPtr(new GpuCommandBuffer())));
            _frame->instancedDynamicPbrMeshes.insert(std::make_pair(cull, GpuCommandBufferPtr(new GpuCommandBuffer())));
            _frame->instancedStaticPbrMeshes.insert(std::make_pair(cull, GpuCommandBufferPtr(new GpuCommandBuffer())));
        }

        // Initialize entity processing
        _entityHandler = INSTANCE(EntityManager)->RegisterEntityProcess<RenderEntityProcess>();

        ClearWorldLight();

        // Copy
        //_prevFrame = std::make_shared<RendererFrame>(*_frame);

        return _renderer->Valid();
    }

    void RendererFrontend::Shutdown() {
        _frame.reset();
        _renderer.reset();

        _entities.clear();
        _dynamicEntities.clear();
        _lights.clear();
        _lightsToRemove.clear();

        INSTANCE(EntityManager)->UnregisterEntityProcess(_entityHandler);
    }

    void RendererFrontend::RecompileShaders() {
        auto ul = _LockWrite();
        _recompileShaders = true;
    }

    void RendererFrontend::_UpdateViewport() {
        _viewportDirty = _viewportDirty || Window::Instance()->WindowResizedWithinLastFrame();
        _frame->viewportDirty = _viewportDirty;

        if (!_viewportDirty) return;
        _viewportDirty = false;

        const float aspect = Window::Instance()->GetWindowDims().first / float(Window::Instance()->GetWindowDims().second);
        _projection        = glm::perspective(
            Radians(_params.fovy).value(),
            aspect,
            _params.znear,
            _params.zfar
        );

        _frame->znear          = _params.znear;
        _frame->zfar           = _params.zfar;
        _frame->projection     = _projection;
        _frame->viewportWidth  = Window::Instance()->GetWindowDims().first;
        _frame->viewportHeight = Window::Instance()->GetWindowDims().second;
        _frame->fovy           = Radians(_params.fovy);
    }

    void RendererFrontend::_UpdateCascadeTransforms() {
        const float cascadeResReciprocal = 1.0f / _frame->csc.cascadeResolutionXY;
        const float cascadeDelta = cascadeResReciprocal;
        const size_t numCascades = _frame->csc.cascades.size();

        _frame->csc.worldLightCamera = CameraPtr(new Camera(false));
        auto worldLightCamera = _frame->csc.worldLightCamera;
        worldLightCamera->setAngle(_worldLight->getRotation());

        // See "Foundations of Game Engine Development, Volume 2: Rendering (pp. 178)
        //
        // FOV_x = 2tan^-1(s/g), FOV_y = 2tan^-1(1/g)
        // ==> tan(FOV_y/2)=1/g ==> g=1/tan(FOV_y/2)
        // where s is the aspect ratio (width / height)

        // Set up the shadow texture offsets
        _frame->csc.cascadeShadowOffsets[0] = glm::vec4(-cascadeDelta, -cascadeDelta, cascadeDelta, -cascadeDelta);
        _frame->csc.cascadeShadowOffsets[1] = glm::vec4(cascadeDelta, cascadeDelta, -cascadeDelta, cascadeDelta);
        // _state.cascadeShadowOffsets[0] = glm::vec4(-cascadeDelta, -cascadeDelta, cascadeDelta, -cascadeDelta);
        // _state.cascadeShadowOffsets[1] = glm::vec4(cascadeDelta, cascadeDelta, -cascadeDelta, cascadeDelta);

        // Assume directional light translation is none
        // Camera light(false);
        // light.setAngle(_state.worldLight.getRotation());
        const Camera & light = *worldLightCamera;
        const Camera & c = *_camera;

        const glm::mat4& lightWorldTransform = light.getWorldTransform();
        const glm::mat4& lightViewTransform = light.getViewTransform();
        const glm::mat4& cameraWorldTransform = c.getWorldTransform();
        const glm::mat4& cameraViewTransform = c.getViewTransform();
        const glm::mat4 transposeLightWorldTransform = glm::transpose(lightWorldTransform);

        // See page 152, eq. 8.21
        const glm::vec3 worldLightDirWorldSpace = -lightWorldTransform[2];
        const glm::vec3 worldLightDirCamSpace = glm::normalize(glm::mat3(cameraViewTransform) * worldLightDirWorldSpace);
        _frame->csc.worldLightDirectionCameraSpace = worldLightDirCamSpace;

        const glm::mat4 L = lightViewTransform * cameraWorldTransform;

        // @see https://gamedev.stackexchange.com/questions/183499/how-do-i-calculate-the-bounding-box-for-an-ortho-matrix-for-cascaded-shadow-mapp
        // @see https://ogldev.org/www/tutorial49/tutorial49.html
        const float ar = float(Window::Instance()->GetWindowDims().first) / float(Window::Instance()->GetWindowDims().second);
        //const float tanHalfHFov = glm::tan(Radians(_params.fovy).value() / 2.0f) * ar;
        //const float tanHalfVFov = glm::tan(Radians(_params.fovy).value() / 2.0f);
        const float projPlaneDist = glm::tan(Radians(_params.fovy).value() / 2.0f);
        const float znear = _params.znear; //0.001f; //_params.znear;
        // We don't want zfar to be unbounded, so we constrain it to at most 800 which also has the nice bonus
        // of increasing our shadow map resolution (same shadow texture resolution over a smaller total area)
        const float zfar  = _params.zfar; //std::min(800.0f, _params.zfar);

        // @see https://johanmedestrom.wordpress.com/2016/03/18/opengl-cascaded-shadow-maps/
        // @see https://johanmedestrom.wordpress.com/2016/03/18/opengl-cascaded-shadow-maps/
        // @see https://developer.download.nvidia.com/SDK/10.5/opengl/src/cascaded_shadow_maps/doc/cascaded_shadow_maps.pdf
        const float lambda = 0.5f;
        const float clipRange = zfar - znear;
        const float ratio = zfar / znear;
        std::vector<float> cascadeEnds(numCascades);
        for (size_t i = 0; i < numCascades; ++i) {
            // We are going to select the cascade split points by computing the logarithmic split, then the uniform split,
            // and then combining them by lambda * log + (1 - lambda) * uniform - the benefit is that it will produce relatively
            // consistent sampling depths over the whole frustum. This is in contrast to under or oversampling inconsistently at different
            // distances.
            const float p = (i + 1) / float(numCascades);
            const float log = znear * std::pow(ratio, p);
            const float uniform = znear + clipRange * p;
            const float d = floorf(lambda * (log - uniform) + uniform);
            cascadeEnds[i] = d;
        }

        // see https://gamedev.stackexchange.com/questions/183499/how-do-i-calculate-the-bounding-box-for-an-ortho-matrix-for-cascaded-shadow-mapp
        // see https://ogldev.org/www/tutorial49/tutorial49.html
        // We offset each cascade begin from 1 onwards so that there is some overlap between the start of cascade k and the end of cascade k-1
        const std::vector<float> cascadeBegins = { 0.0f, cascadeEnds[0] - 10.0f,  cascadeEnds[1] - 10.0f, cascadeEnds[2] - 10.0f }; // 4 cascades max
        //const std::vector<float> cascadeEnds   = {  30.0f, 100.0f, 240.0f, 640.0f };
        std::vector<float> aks;
        std::vector<float> bks;
        std::vector<float> dks;
        std::vector<glm::vec3> sks;
        std::vector<float> zmins;
        std::vector<float> zmaxs;

        for (size_t i = 0; i < numCascades; ++i) {
            const float ak = cascadeBegins[i];
            const float bk = cascadeEnds[i];
            _frame->csc.cascades[i].cascadeBegins = ak;
            _frame->csc.cascades[i].cascadeEnds   = bk;
            aks.push_back(ak);
            bks.push_back(bk);

            // These base values are in camera space and define our frustum corners
            const float xn = ak * ar * projPlaneDist;
            const float xf = bk * ar * projPlaneDist;
            const float yn = ak * projPlaneDist;
            const float yf = bk * projPlaneDist;
            // Keep all of these in camera space for now
            std::vector<glm::vec4> frustumCorners = {
                // Near corners
                glm::vec4(xn, yn, -ak, 1.0f),
                glm::vec4(-xn, yn, -ak, 1.0f),
                glm::vec4(xn, -yn, -ak, 1.0f),
                glm::vec4(-xn, -yn, -ak, 1.0f),

                // Far corners
                glm::vec4(xf, yf, -bk, 1.0f),
                glm::vec4(-xf, yf, -bk, 1.0f),
                glm::vec4(xf, -yf, -bk, 1.0f),
                glm::vec4(-xf, -yf, -bk, 1.0f),
            };

            // Calculate frustum center
            // @see https://ahbejarano.gitbook.io/lwjglgamedev/chapter26
            glm::vec3 frustumSum(0.0f);
            for (auto& v : frustumCorners) frustumSum += glm::vec3(v);
            const glm::vec3 frustumCenter = frustumSum / float(frustumCorners.size());

            // Calculate max diameter across frustum
            float maxLength = std::numeric_limits<float>::min();
            for (int i = 0; i < frustumCorners.size() - 1; ++i) {
                for (int j = 1; j < frustumCorners.size(); ++j) {
                    maxLength = std::max<float>(maxLength, glm::length(frustumCorners[i] - frustumCorners[j]));
                }
            }
            
            // This tells us the maximum diameter for the cascade bounding box
            //const float dk = std::ceilf(std::max<float>(glm::length(frustumCorners[0] - frustumCorners[6]), 
            //                                            glm::length(frustumCorners[4] - frustumCorners[6])));
            const float dk = ceilf(maxLength);
            dks.push_back(dk);
            // T is essentially the physical width/height of area corresponding to each texel in the shadow map
            const float T = dk / _frame->csc.cascadeResolutionXY;
            _frame->csc.cascades[i].cascadeRadius = dk / 2.0f;

            // Compute min/max of each so that we can combine it with dk to create a perfectly rectangular bounding box
            glm::vec3 minVec;
            glm::vec3 maxVec;
            for (int j = 0; j < frustumCorners.size(); ++j) {
                // First make sure to transform frustumCorners[j] from camera space to light space
                frustumCorners[j] = L * frustumCorners[j];
                const glm::vec3 frustumVec = frustumCorners[j];
                if (j == 0) {
                    minVec = frustumVec;
                    maxVec = frustumVec;
                }
                else {
                    minVec = glm::min(minVec, frustumVec);
                    maxVec = glm::max(maxVec, frustumVec);
                }
            }

            const float minX = minVec.x;
            const float maxX = maxVec.x;

            const float minY = minVec.y;
            const float maxY = maxVec.y;

            const float minZ = minVec.z;
            const float maxZ = maxVec.z;

            zmins.push_back(minZ);
            zmaxs.push_back(maxZ);

            // Now we calculate cascade camera position sk using the min, max, dk and T for a stable location
            glm::vec3 sk(floorf((maxX + minX) / (2.0f * T)) * T, 
                         floorf((maxY + minY) / (2.0f * T)) * T, 
                         minZ);
            //sk = glm::vec3(L * glm::vec4(sk, 1.0f));
            // STRATUS_LOG << "sk " << sk << std::endl;
            sks.push_back(sk);
            _frame->csc.cascades[i].cascadePositionLightSpace = sk;
            _frame->csc.cascades[i].cascadePositionCameraSpace = glm::vec3(cameraViewTransform * lightWorldTransform * glm::vec4(sk, 1.0f));

            // We use transposeLightWorldTransform because it's less precision-error-prone than just doing glm::inverse(lightWorldTransform)
            // Note: we use -sk instead of lightWorldTransform * sk because we're assuming the translation component is 0
            const glm::mat4 cascadeViewTransform = glm::mat4(transposeLightWorldTransform[0], 
                                                            transposeLightWorldTransform[1],
                                                            transposeLightWorldTransform[2],
                                                            glm::vec4(-sk, 1.0f));

            // We add this into the cascadeOrthoProjection map to add a slight depth offset to each value which helps reduce flickering artifacts
            const float shadowDepthOffset = 0.0f;//2e-19;
            // We are putting the light camera location sk on the near plane in the halfway point between left, right, top and bottom planes
            // so it enables us to use the simplified Orthographic Projection matrix below
            //
            // This results in values between [-1, 1]
            const glm::mat4 cascadeOrthoProjection(glm::vec4(2.0f / dk, 0.0f, 0.0f, 0.0f), 
                                                   glm::vec4(0.0f, 2.0f / dk, 0.0f, 0.0f),
                                                   glm::vec4(0.0f, 0.0f, 1.0f / (maxZ - minZ), shadowDepthOffset),
                                                   glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
            //const glm::mat4 cascadeOrthoProjection(glm::vec4(2.0f / (maxX - minX), 0.0f, 0.0f, 0.0f), 
            //                                       glm::vec4(0.0f, 2.0f / (maxY - minY), 0.0f, 0.0f),
            //                                       glm::vec4(0.0f, 0.0f, 1.0f / (maxZ - minZ), shadowDepthOffset),
            //                                       glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));

            // // // Gives us x, y values between [0, 1]
            //const glm::mat4 cascadeTexelOrthoProjection(glm::vec4(1.0f / dk, 0.0f, 0.0f, 0.0f), 
            //                                            glm::vec4(0.0f, 1.0f / dk, 0.0f, 0.0f),
            //                                            glm::vec4(0.0f, 0.0f, 1.0f / (maxZ - minZ), 0.0f),
            //                                            glm::vec4(0.5f, 0.5f, 0.5f, 1.0f));
            const glm::mat4 cascadeTexelOrthoProjection = cascadeOrthoProjection;

            // Note: if we want we can set texelProjection to be cascadeTexelOrthoProjection and then set projectionView
            // to be cascadeTexelOrthoProjection * cascadeViewTransform. This has the added benefit of automatically translating
            // x, y positions to texel coordinates on the range [0, 1] rather than [-1, 1].
            //
            // However, the alternative is to just compute (coordinate * 0.5 + 0.5) in the fragment shader which does the same thing.
            _frame->csc.cascades[i].projectionViewRender = cascadeOrthoProjection * cascadeViewTransform;
            _frame->csc.cascades[i].projectionViewSample = cascadeTexelOrthoProjection * cascadeViewTransform;
            //STRATUS_LOG << _frame->csc.cascades[i].projectionViewSample << std::endl;

            if (i > 0) {
                // See page 187, eq. 8.82
                // Ck = Mk_shadow * (M0_shadow) ^ -1
                glm::mat4 Ck = _frame->csc.cascades[i].projectionViewSample * glm::inverse(_frame->csc.cascades[0].projectionViewSample);
                _frame->csc.cascades[i].sampleCascade0ToCurrent = Ck;

                // This will allow us to calculate the cascade blending weights in the vertex shader and then
                // the cascade indices in the pixel shader
                const glm::vec3 n = -glm::vec3(cameraWorldTransform[2]);
                const glm::vec3 c = glm::vec3(cameraWorldTransform[3]);
                // fk now represents a plane along the direction of the view frustum. Its normal is equal to the camera's forward
                // direction in world space and it contains the point c + ak*n.
                const glm::vec4 fk = glm::vec4(n.x, n.y, n.z, glm::dot(-n, c) - ak) * (1.0f / (bks[i - 1] - ak));
                _frame->csc.cascades[i].cascadePlane = fk;
                //STRATUS_LOG << fk << std::endl;
                //_frame->csc.cascades[i].cascadePlane = glm::vec4(10.0f);
            }
        }
    }

    bool RendererFrontend::_EntityChanged(const EntityPtr& p) {
        auto tc = p->Components().GetComponent<GlobalTransformComponent>().component;
        auto rc = p->Components().GetComponent<RenderComponent>().component;
        return tc->ChangedWithinLastFrame() || rc->ChangedWithinLastFrame();
    }

    void RendererFrontend::_CheckEntitySetForChanges(std::unordered_set<EntityPtr>& set) {
        for (auto& entity : set) {
            // If this is a light-interacting node, run through all the lights to see if they need to be updated
            if (_EntityChanged(entity)) {               

                InitializeMeshTransformComponent(entity);

                _drawCommandsDirty = true;

                if (IsLightInteracting(entity)) {
                    for (const auto& light : _lights) {
                        // Static lights don't care about entity movement changes
                        if (light->IsStaticLight()) continue;

                        auto lightPos = light->GetPosition();
                        auto lightRadius = light->getRadius();
                        //If the EntityView is in the light's visible set, its shadows are now out of date
                        for (size_t i = 0; i < GetMeshCount(entity); ++i) {
                            if (glm::distance(GetWorldTransform(entity, i), lightPos) > lightRadius) {
                                _frame->lightsToUpate.PushBack(light);
                            }
                            // If the EntityView has moved inside the light's radius, add it
                            else if (glm::distance(GetWorldTransform(entity, i), lightPos) < lightRadius) {
                                _frame->lightsToUpate.PushBack(light);
                            }
                        }
                    }
                }
            }
        }
    }

    void RendererFrontend::_CheckForEntityChanges() {
        // We only care about dynamic light-interacting entities
        _CheckEntitySetForChanges(_dynamicEntities);
    }

    void RendererFrontend::_UpdateLights() {
        _frame->lightsToRemove.clear();
        // First get rid of all lights that are pending deletion
        for (auto& light : _lightsToRemove) {
            _frame->lights.erase(light);
            _frame->virtualPointLights.erase(light);
            _frame->lightsToRemove.insert(light);
        }
        _lightsToRemove.clear();

        // Update the world light
        _frame->csc.worldLight = _worldLight;//->Copy();

        // Now go through and update all lights that have changed in some way
        for (auto& light : _lights) {
            if ( !light->castsShadows() ) continue;

            // See if the light moved or its radius changed
            if (light->PositionChangedWithinLastFrame() || light->RadiusChangedWithinLastFrame()) {
                _frame->lightsToUpate.PushBack(light);
            }
        }
    }

    static bool ValidateTexture(const Async<Texture> & tex) {
        return tex.Completed() && !tex.Failed();
    }

    void RendererFrontend::_RecalculateMaterialSet() {
        _frame->materialInfo.availableMaterials.clear();
        _materialsDirty = true;

        for (const EntityPtr& p : _entities) {
            _AddAllMaterialsForEntity(p);
        }

        // After this loop anything left in indices is no longer referenced
        // by an entity
        for (const MaterialPtr& m : _frame->materialInfo.availableMaterials) {
            _frame->materialInfo.indices.erase(m);
        }

    #define MAKE_NON_RESIDENT(handle)                               \
        {                                                           \
        auto at = INSTANCE(ResourceManager)->LookupTexture(handle); \
        if (ValidateTexture(at)) {                                  \
            Texture::MakeNonResident(at.Get());                     \
        }                                                           \
        }

        // Erase what is no longer referenced
        for (auto& entry : _frame->materialInfo.indices) {
            MaterialPtr material = entry.first;
            MAKE_NON_RESIDENT(material->GetDiffuseTexture())
            MAKE_NON_RESIDENT(material->GetAmbientTexture())
            MAKE_NON_RESIDENT(material->GetNormalMap())
            MAKE_NON_RESIDENT(material->GetDepthMap())
            MAKE_NON_RESIDENT(material->GetRoughnessMap())
            MAKE_NON_RESIDENT(material->GetMetallicMap())
            MAKE_NON_RESIDENT(material->GetMetallicRoughnessMap())
        }

        _frame->materialInfo.indices.clear();
    
    #undef MAKE_NON_RESIDENT
    }

    void RendererFrontend::_CopyMaterialToGpuAndMarkForUse(const MaterialPtr& material, GpuMaterial* gpuMaterial) {
        gpuMaterial->flags = 0;

        gpuMaterial->diffuseColor = material->GetDiffuseColor();
        gpuMaterial->ambientColor = glm::vec4(material->GetAmbientColor(), 1.0f);
        gpuMaterial->baseReflectivity = glm::vec4(material->GetBaseReflectivity(), 1.0f);
        gpuMaterial->metallicRoughness = glm::vec4(material->GetMetallic(), material->GetRoughness(), 0.0f, 0.0f);

        auto diffuseHandle =   material->GetDiffuseTexture();
        auto ambientHandle =   material->GetAmbientTexture();
        auto normalHandle =    material->GetNormalMap();
        auto depthHandle =     material->GetDepthMap();
        auto roughnessHandle = material->GetRoughnessMap();
        auto metallicHandle =  material->GetMetallicMap();
        auto metallicRoughnessHandle = material->GetMetallicRoughnessMap();
        
        auto diffuse = INSTANCE(ResourceManager)->LookupTexture(diffuseHandle);
        auto ambient = INSTANCE(ResourceManager)->LookupTexture(ambientHandle);
        auto normal = INSTANCE(ResourceManager)->LookupTexture(normalHandle);
        auto depth = INSTANCE(ResourceManager)->LookupTexture(depthHandle);
        auto roughness = INSTANCE(ResourceManager)->LookupTexture(roughnessHandle);
        auto metallic = INSTANCE(ResourceManager)->LookupTexture(metallicHandle);
        auto metallicRoughness = INSTANCE(ResourceManager)->LookupTexture(metallicRoughnessHandle);

        if (ValidateTexture(diffuse)) {
            gpuMaterial->diffuseMap = diffuse.Get().GpuHandle();
            gpuMaterial->flags |= GPU_DIFFUSE_MAPPED;
            Texture::MakeResident(diffuse.Get());
        }
        // If this is true then the texture is still loading so we need to check again later
        else if (diffuseHandle != TextureHandle::Null()) {
            _materialsDirty = true;
        }

        if (ValidateTexture(ambient)) {
            gpuMaterial->ambientMap = ambient.Get().GpuHandle();
            gpuMaterial->flags |= GPU_AMBIENT_MAPPED;
            Texture::MakeResident(ambient.Get());
        }
        // If this is true then the texture is still loading so we need to check again later
        else if (ambientHandle != TextureHandle::Null()) {
            _materialsDirty = true;
        }

        if (ValidateTexture(normal)) {
            gpuMaterial->normalMap = normal.Get().GpuHandle();
            gpuMaterial->flags |= GPU_NORMAL_MAPPED;
            Texture::MakeResident(normal.Get());
        }
        // If this is true then the texture is still loading so we need to check again later
        else if (normalHandle != TextureHandle::Null()) {
            _materialsDirty = true;
        }

        if (ValidateTexture(depth)) {
            gpuMaterial->depthMap = depth.Get().GpuHandle();
            gpuMaterial->flags |= GPU_DEPTH_MAPPED;       
            Texture::MakeResident(depth.Get());
        }
        // If this is true then the texture is still loading so we need to check again later
        else if (depthHandle != TextureHandle::Null()) {
            _materialsDirty = true;
        }

        if (ValidateTexture(roughness)) {
            gpuMaterial->roughnessMap = roughness.Get().GpuHandle();
            gpuMaterial->flags |= GPU_ROUGHNESS_MAPPED;
            Texture::MakeResident(roughness.Get());
        }
        // If this is true then the texture is still loading so we need to check again later
        else if (roughnessHandle != TextureHandle::Null()) {
            _materialsDirty = true;
        }

        if (ValidateTexture(metallic)) {
            gpuMaterial->metallicMap = metallic.Get().GpuHandle();
            gpuMaterial->flags |= GPU_METALLIC_MAPPED;
            Texture::MakeResident(metallic.Get());
        }
        // If this is true then the texture is still loading so we need to check again later
        else if (metallicHandle != TextureHandle::Null()) {
            _materialsDirty = true;
        }

        if (ValidateTexture(metallicRoughness)) {
            gpuMaterial->metallicRoughnessMap = metallicRoughness.Get().GpuHandle();
            gpuMaterial->flags |= GPU_METALLIC_ROUGHNESS_MAPPED;
            Texture::MakeResident(metallicRoughness.Get());
        }
        // If this is true then the texture is still loading so we need to check again later
        else if (metallicRoughnessHandle != TextureHandle::Null()) {
            _materialsDirty = true;
        }
    }

    void RendererFrontend::_UpdateMaterialSet() {
        // static int frame = 0;
        // ++frame;
        // if (frame % 10 == 0) _materialsDirty = true;

        // See if any materials were changed within the last frame
        if ( !_materialsDirty ) {
            for (auto& entry : _frame->materialInfo.indices) {
                if (entry.first->ChangedWithinLastFrame()) {
                    _materialsDirty = true;
                    break;
                }
            }
        }

        // Update for the next 3 frames after the frame indices were recalculated (solved a strange performance issue possible related to shaders
        // accessing invalid data)
        const uint64_t frameCount = INSTANCE(Engine)->FrameCount();
        const bool updateMaterialResidency = _materialsDirty || ((frameCount - _lastFrameMaterialIndicesRecomputed) < 3);

        // If no materials to update then no need to recompute the index set
        if (_materialsDirty) {
            _drawCommandsDirty = true;
            _materialsDirty = false;
            _lastFrameMaterialIndicesRecomputed = frameCount;

            if (_frame->materialInfo.availableMaterials.size() >= _frame->materialInfo.maxMaterials) {
                throw std::runtime_error("Maximum number of materials exceeded");
            }

            std::unordered_map<MaterialPtr, uint32_t>& indices = _frame->materialInfo.indices;
            indices.clear();

            for (auto material : _frame->materialInfo.availableMaterials) {
                int nextIndex = int(indices.size());
                indices.insert(std::make_pair(material, nextIndex));
            }
        }

        // If we don't update these either every frame or every few frames, performance degrades. I do not yet know 
        // why this happens.
        if (updateMaterialResidency) {
            if (_frame->materialInfo.materials.size() < _frame->materialInfo.availableMaterials.size()) {
                _frame->materialInfo.materials.resize(_frame->materialInfo.availableMaterials.size());
            }

            for (const auto& entry : _frame->materialInfo.indices) {
                GpuMaterial * material = &_frame->materialInfo.materials[entry.second];
                _CopyMaterialToGpuAndMarkForUse(entry.first, material);
            }

            _frame->materialInfo.materialsBuffer.CopyDataToBuffer(0, 
                                                                sizeof(GpuMaterial) * _frame->materialInfo.materials.size(),
                                                                (const void *)_frame->materialInfo.materials.data());
        }
    }

    std::unordered_map<RenderFaceCulling, std::vector<GpuDrawElementsIndirectCommand>> RendererFrontend::_GenerateDrawCommands(RenderComponent * c) const {
        std::unordered_map<RenderFaceCulling, std::vector<GpuDrawElementsIndirectCommand>> commands;
        for (size_t i = 0; i < c->GetMeshCount(); ++i) {
            auto cull = c->GetMesh(i)->GetFaceCulling();
            if (commands.find(cull) == commands.end()) {
                auto vec = std::vector<GpuDrawElementsIndirectCommand>();
                vec.reserve(c->GetMeshCount());
                commands.insert(std::make_pair(cull, std::move(vec)));
            }
            GpuDrawElementsIndirectCommand command;
            auto& commandList = commands.find(cull)->second;
            command.baseInstance = 0;
            command.baseVertex = 0;
            command.firstIndex = c->GetMesh(i)->GetIndexOffset();
            command.instanceCount = 1;
            command.vertexCount = c->GetMesh(i)->GetNumIndices();
            commandList.push_back(command);
        }
        return commands;
    }

    void RendererFrontend::_UpdateDrawCommands() {
        if (!_drawCommandsDirty) return;
        _drawCommandsDirty = false;

        // Clear old commands
        std::vector<std::unordered_map<RenderFaceCulling, GpuCommandBufferPtr> *> oldCommands{
            &_frame->instancedFlatMeshes,
            &_frame->instancedDynamicPbrMeshes,
            &_frame->instancedStaticPbrMeshes
        };

        for (auto * cmdList : oldCommands) {
            for (auto& entry : *cmdList) {
                entry.second->materialIndices.clear();
                entry.second->modelTransforms.clear();
                entry.second->indirectDrawCommands.clear();
            }
        }

    #define GENERATE_COMMANDS(entityMap, drawCommands)                                                                 \
        for (const auto& entry : entityMap) {                                                                          \
            RenderComponent * c = GetComponent<RenderComponent>(entry.first);                                          \
            MeshWorldTransforms * mt = GetComponent<MeshWorldTransforms>(entry.first);                                 \
            auto commands = _GenerateDrawCommands(c);                                                                  \
            for (size_t i = 0; i < c->GetMeshCount(); ++i) {                                                           \
                auto cull = c->GetMesh(i)->GetFaceCulling();                                                           \
                auto& buffer = drawCommands.find(cull)->second;                                                        \
                buffer->materialIndices.push_back(_frame->materialInfo.indices.find(c->GetMaterialAt(i))->second);     \
                buffer->modelTransforms.push_back(mt->transforms[i]);                                                  \
            }                                                                                                          \
            for (auto& entry : commands) {                                                                             \
                auto& buffer = drawCommands.find(entry.first)->second;                                                 \
                buffer->indirectDrawCommands.insert(buffer->indirectDrawCommands.end(),                                \
                                                    entry.second.begin(), entry.second.end());                         \
            }                                                                                                          \
        }                                                                                                              \
        for (auto& entry : drawCommands) {                                                                             \
            entry.second->UploadDataToGpu();                                                                           \
        }

        // Generate flat commands
        GENERATE_COMMANDS(_flatEntities, _frame->instancedFlatMeshes)

        // Generate pbr commands
        GENERATE_COMMANDS(_dynamicPbrEntities, _frame->instancedDynamicPbrMeshes)
        GENERATE_COMMANDS(_staticPbrEntities, _frame->instancedStaticPbrMeshes)

    #undef GENERATE_COMMANDS
    }
}