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
#include "StratusGpuMaterialBuffer.h"

#include <algorithm>

namespace stratus {
    using Vec3Allocator = StackBasedPoolAllocator<glm::vec3>;
    using Vec4Allocator = StackBasedPoolAllocator<glm::vec4>;
    using Mat3Allocator = StackBasedPoolAllocator<glm::mat3>;
    using Mat4Allocator = StackBasedPoolAllocator<glm::mat4>;

    struct RenderEntityProcess : public EntityProcess {
        virtual ~RenderEntityProcess() = default;

        virtual void Process(const double deltaSeconds) {}

        void EntitiesAdded(const std::unordered_set<stratus::EntityPtr>& e) override {
            auto rf = INSTANCE(RendererFrontend);
            if (rf) rf->EntitiesAdded_(e);
        }

        void EntitiesRemoved(const std::unordered_set<stratus::EntityPtr>& e) override {
            auto rf = INSTANCE(RendererFrontend);
            if (rf) rf->EntitiesRemoved_(e);
        }

        void EntityComponentsAdded(const std::unordered_map<stratus::EntityPtr, std::vector<stratus::EntityComponent *>>& e) override {
            auto rf = INSTANCE(RendererFrontend);
            if (rf) rf->EntityComponentsAdded_(e);
        }

        void EntityComponentsEnabledDisabled(const std::unordered_set<stratus::EntityPtr>& e) override {
            auto rf = INSTANCE(RendererFrontend);
            if (rf) rf->EntityComponentsEnabledDisabled_(e);
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
        : params_(p) {
    }

    void RendererFrontend::AddAllMaterialsForEntity_(const EntityPtr& p) {
        RenderComponent * c = p->Components().GetComponent<RenderComponent>().component;
        frame_->materialInfo->MarkMaterialsUsed(c);
    }

    void RendererFrontend::RemoveAllMaterialsForEntity_(const EntityPtr& p) {
        RenderComponent* c = p->Components().GetComponent<RenderComponent>().component;
        frame_->materialInfo->MarkMaterialsUnused(c);
    }

    void RendererFrontend::EntitiesAdded_(const std::unordered_set<stratus::EntityPtr>& e) {
        auto ul = LockWrite_();
        bool added = false;
        for (auto ptr : e) {
            added |= AddEntity_(ptr);
        }
    }

    void RendererFrontend::EntitiesRemoved_(const std::unordered_set<stratus::EntityPtr>& e) {
        auto ul = LockWrite_();
        bool removed = false;
        for (auto& ptr : e) {
            removed = removed || RemoveEntity_(ptr);
        }
    }

    void RendererFrontend::EntityComponentsAdded_(const std::unordered_map<stratus::EntityPtr, std::vector<stratus::EntityComponent *>>& e) {
        auto ul = LockWrite_();
        bool changed = false;
        for (auto& entry : e) {
            auto ptr = entry.first;
            if (RemoveEntity_(ptr)) {
                changed = true;
                AddEntity_(ptr);
            }
        }
    }

    void RendererFrontend::EntityComponentsEnabledDisabled_(const std::unordered_set<stratus::EntityPtr>& e) {
        auto ul = LockWrite_();
        bool changed = false;
        for (auto& ptr : e) {
            if (RemoveEntity_(ptr)) {
                changed = true;
                AddEntity_(ptr);
            }
        }
    }

    bool RendererFrontend::AddEntity_(const EntityPtr& p) {
        if (p == nullptr || entities_.find(p) != entities_.end()) return false;
        
        if (IsRenderable(p)) {
            InitializeMeshTransformComponent(p);

            entities_.insert(p);

            const bool isStatic = IsStaticEntity(p);

            if (!isStatic) {
                dynamicEntities_.insert(p);
            }

            AddAllMaterialsForEntity_(p);
            
            frame_->drawCommands->RecordCommands(p, frame_->materialInfo);
            //_renderComponents.insert(p->Components().GetComponent<RenderComponent>().component);
            
            if (IsLightInteracting(p)) {
                for (size_t i = 0; i < GetMeshCount(p); ++i) {
                    if (isStatic) InsertMesh(staticPbrEntities_, p, i);
                    else InsertMesh(dynamicPbrEntities_, p, i);

                    for (auto& entry : lights_) {
                        if (!entry->CastsShadows()) continue;
                        auto pos = entry->GetPosition();
                        if ((isStatic && entry->IsStaticLight()) || !entry->IsStaticLight()) {
                            if (glm::distance(GetWorldTransform(p, i), pos) < entry->GetRadius()) {
                                frame_->lightsToUpdate.PushBack(entry);
                            }
                        }
                    }
                }
            }
            else {
                for (size_t i = 0; i < GetMeshCount(p); ++i) {
                    InsertMesh(flatEntities_, p, i);
                }
            }

            return true;
        }
        else {
            return false;
        }
    }

    bool RendererFrontend::RemoveEntity_(const EntityPtr& p) {
        if (p == nullptr || entities_.find(p) == entities_.end() || !IsRenderable(p)) return false;

        entities_.erase(p);
        dynamicEntities_.erase(p);
        dynamicPbrEntities_.erase(p);
        staticPbrEntities_.erase(p);
        flatEntities_.erase(p);

        RemoveAllMaterialsForEntity_(p);

        frame_->drawCommands->RemoveAllCommands(p);

        const auto entityIsStatic = IsStaticEntity(p);

        for (auto& entry : lights_) {
            if (!entry->CastsShadows()) continue;
            //if (entry.second.visible.erase(p)) {
                if (entry->IsStaticLight()) {
                    if (entityIsStatic) {
                        frame_->lightsToUpdate.PushBack(entry);
                    }
                }
                else {
                    frame_->lightsToUpdate.PushBack(entry);
                }
            //}
        }

        return true;
    }

    void RendererFrontend::AddLight(const LightPtr& light) {
        auto ul = LockWrite_();
        if (lights_.find(light) != lights_.end()) return;

        lights_.insert(light);
        frame_->lights.insert(light);

        if ( light->IsVirtualLight() ) virtualPointLights_.insert(light);

        if ( light->IsVirtualLight() || light->IsStaticLight() ) {
            staticLights_.insert(light);
        }
        else {
            dynamicLights_.insert(light);
        }

        if ( !light->CastsShadows() ) return;

        frame_->lightsToUpdate.PushBack(light);

        //_AttemptAddEntitiesForLight(light, data, _frame->instancedPbrMeshes);
    }

    void RendererFrontend::RemoveLight(const LightPtr& light) {
        auto ul = LockWrite_();
        if (lights_.find(light) == lights_.end()) return;
        lights_.erase(light);
        dynamicLights_.erase(light);
        staticLights_.erase(light);
        virtualPointLights_.erase(light);
        lightsToRemove_.insert(light);
        frame_->lightsToUpdate.Erase(light);
    }

    void RendererFrontend::ClearLights() {
        auto ul = LockWrite_();
        for (auto& light : lights_) {
            lightsToRemove_.insert(light);
        }
        lights_.clear();
        dynamicLights_.clear();
        staticLights_.clear();
        virtualPointLights_.clear();
        frame_->lightsToUpdate.Clear();
    }

    void RendererFrontend::SetWorldLight(const InfiniteLightPtr& light) {
        if (light == nullptr) return;
        auto ul = LockWrite_();
        worldLight_ = light;
    }

    InfiniteLightPtr RendererFrontend::GetWorldLight() {
        auto sl = LockRead_();
        return worldLight_;
    }

    void RendererFrontend::ClearWorldLight() {
        auto ul = LockWrite_();
        // Create a dummy world light that is disabled
        worldLight_ = InfiniteLightPtr(new InfiniteLight(false));
    }

    void RendererFrontend::SetCamera(const CameraPtr& camera) {
        auto ul = LockWrite_();
        camera_ = camera;
    }

    CameraPtr RendererFrontend::GetCamera() const {
        auto sl = LockRead_();
        return camera_;
    }

    void RendererFrontend::SetFovY(const Degrees& fovy) {
        auto ul = LockWrite_();
        params_.fovy = fovy;
        viewportDirty_ = true;
    }

    void RendererFrontend::SetNearFar(const float znear, const float zfar) {
        auto ul = LockWrite_();
        params_.znear = znear;
        params_.zfar  = zfar;
        viewportDirty_ = true;
    }

    void RendererFrontend::SetClearColor(const glm::vec4& color) {
        auto ul = LockWrite_();
        frame_->clearColor = color;
    }

    RendererSettings RendererFrontend::GetSettings() const {
        auto sl = LockRead_();
        return frame_->settings;
    }

    void RendererFrontend::SetSettings(const RendererSettings& settings) {
        auto ul = LockWrite_();
        frame_->settings = settings;
    }

    static glm::vec2 GetJitterForIndex(const size_t index, const float width, const float height) {
        glm::vec2 jitter(haltonSequence[index].first, haltonSequence[index].second);
        // Halton numbers are from [0, 1] so we convert this to an appropriate +/- subpixel offset
        //jitter = ((jitter - glm::vec2(0.5f)) / glm::vec2(width, height)) * 2.0f;
        // Convert from [0, 1] to [-0.5, 0.5]
        jitter = jitter - 0.5f;
        // Scale to appropriate subpixel size by using viewport width/height
        jitter = jitter / glm::vec2(width, height);

        return jitter;
    }

    SystemStatus RendererFrontend::Update(const double deltaSeconds) {
        CHECK_IS_APPLICATION_THREAD();

        auto ul = LockWrite_();
        if (camera_ == nullptr) return SystemStatus::SYSTEM_CONTINUE;

        // Update per frame scratch memory if application requested a different size
        if (frame_->settings.perFrameMaxScratchMemoryBytes > 0 &&
            frame_->perFrameScratchMemory->Capacity() < frame_->settings.perFrameMaxScratchMemoryBytes) {
            STRATUS_LOG << "Resizing per frame scratch memory for renderer to " << frame_->settings.perFrameMaxScratchMemoryBytes;
            frame_->perFrameScratchMemory = MakeUnsafe<StackAllocator>(frame_->settings.perFrameMaxScratchMemoryBytes);
        }

        camera_->Update(deltaSeconds);
        frame_->camera = camera_->Copy();
        frame_->view = camera_->GetViewTransform();

        UpdateViewport_();
        UpdateCascadeData_();
        CheckForEntityChanges_();
        UpdateLights_();
        UpdateMaterialSet_();
        UpdateDrawCommands_();
        UpdateVisibility_();

        // Update view projection and its inverse
        frame_->projectionView = frame_->projection * frame_->view;
        frame_->invProjectionView = glm::inverse(frame_->projectionView);

        // Increment halton index - only use a max of the first 16 samples
        const size_t maxIndex = std::min<size_t>(16, haltonSequence.size());
        currentHaltonIndex_ = (currentHaltonIndex_ + 1) % maxIndex;

        // Set up the jittered variant
        glm::vec2 jitter(0.0f);
        if (frame_->settings.taaEnabled) {
            jitter = GetJitterForIndex(currentHaltonIndex_, float(frame_->viewportWidth), float(frame_->viewportHeight));
        }
        frame_->jitterProjectionView = frame_->projection;
        frame_->jitterProjectionView[3][0] += jitter.x;
        frame_->jitterProjectionView[3][1] += jitter.y;
        frame_->jitterProjectionView = frame_->jitterProjectionView * frame_->view;

        //_SwapFrames();

        // Check for shader recompile request
        if (recompileShaders_) {
            renderer_->RecompileShaders();
            viscullLodSelect_->Recompile();
            viscull_->Recompile();
            recompileShaders_ = false;
        }

        // Begin the new frame
        renderer_->Begin(frame_, true);

        // Complete the frame
        renderer_->RenderScene(deltaSeconds);
        renderer_->End();

        // This needs to be unset
        frame_->csc.regenerateFbo = false;

        // Move current transforms -> previous transforms
        UpdatePrevFrameModelTransforms_();

        // Set previous projection view
        frame_->prevProjectionView = frame_->projectionView;

        // Reset the per frame scratch memory
        frame_->perFrameScratchMemory->Deallocate();

        return SystemStatus::SYSTEM_CONTINUE;
    }

    bool RendererFrontend::Initialize() {
        CHECK_IS_APPLICATION_THREAD();
        // Create the renderer on the renderer thread only
        renderer_ = std::make_unique<RendererBackend>(Window::Instance()->GetWindowDims().first, Window::Instance()->GetWindowDims().second, params_.appName);

        frame_ = std::make_shared<RendererFrame>();

        // 4 cascades total
        frame_->csc.cascades.resize(4);
        frame_->csc.cascadeResolutionXY = 1024;
        frame_->csc.regenerateFbo = true;

        // Set materials per frame and initialize material buffer
        frame_->materialInfo = GpuMaterialBuffer::Create(8192);

        // Initialize per frame scratch memory
        frame_->perFrameScratchMemory = MakeUnsafe<StackAllocator>(frame_->settings.perFrameMaxScratchMemoryBytes);

        //_frame->instancedFlatMeshes.resize(1);
        //_frame->instancedDynamicPbrMeshes.resize(1);
        //_frame->instancedStaticPbrMeshes.resize(1);

        // Set up draw command buffers
        frame_->drawCommands = GpuCommandManager::Create(8);

        // Initialize entity processing
        entityHandler_ = INSTANCE(EntityManager)->RegisterEntityProcess<RenderEntityProcess>();

        ClearWorldLight();

        // Initialize visibility culling compute pipeline
        const std::filesystem::path shaderRoot("../Source/Shaders");
        const ShaderApiVersion version{GraphicsDriver::GetConfig().majorVersion, GraphicsDriver::GetConfig().minorVersion};

        viscullLodSelect_ = std::unique_ptr<Pipeline>(new Pipeline(shaderRoot, version, {
            Shader{"visibility_culling.cs", ShaderType::COMPUTE}},
            // Defines
            { {"SELECT_LOD", "1"} }
        ));

        viscull_ = std::unique_ptr<Pipeline>(new Pipeline(shaderRoot, version, {
            Shader{"visibility_culling.cs", ShaderType::COMPUTE} }
        ));

        updateTransforms_ = std::unique_ptr<Pipeline>(new Pipeline(shaderRoot, version, {
            Shader{"update_model_transforms.cs", ShaderType::COMPUTE} }
        ));

        // Copy
        //_prevFrame = std::make_shared<RendererFrame>(*_frame);

        return renderer_->Valid() && viscullLodSelect_->IsValid() && viscull_->IsValid() && updateTransforms_->IsValid();
    }

    void RendererFrontend::Shutdown() {
        frame_.reset();
        renderer_.reset();

        entities_.clear();
        dynamicEntities_.clear();
        lights_.clear();
        lightsToRemove_.clear();

        INSTANCE(EntityManager)->UnregisterEntityProcess(entityHandler_);
    }

    void RendererFrontend::RecompileShaders() {
        auto ul = LockWrite_();
        recompileShaders_ = true;
    }

    void RendererFrontend::UpdateViewport_() {
        viewportDirty_ = viewportDirty_ || Window::Instance()->WindowResizedWithinLastFrame();
        frame_->viewportDirty = viewportDirty_;

        if (!viewportDirty_) return;
        viewportDirty_ = false;

        const float aspect = float(Window::Instance()->GetWindowDims().first) / float(Window::Instance()->GetWindowDims().second);
        projection_        = glm::perspective(
            Radians(params_.fovy).value(),
            aspect,
            params_.znear,
            params_.zfar
        );

        frame_->znear          = params_.znear;
        frame_->zfar           = params_.zfar;
        frame_->projection     = projection_;
        frame_->viewportWidth  = Window::Instance()->GetWindowDims().first;
        frame_->viewportHeight = Window::Instance()->GetWindowDims().second;
        frame_->fovy           = Radians(params_.fovy);
    }

    void RendererFrontend::UpdateCascadeData_() {
        auto requestedCascadeResolutionXY = static_cast<uint32_t>(frame_->settings.cascadeResolution);

        frame_->csc.regenerateFbo = frame_->csc.cascadeResolutionXY != requestedCascadeResolutionXY;

        frame_->csc.cascadeResolutionXY = requestedCascadeResolutionXY;

        const float cascadeResReciprocal = 1.0f / frame_->csc.cascadeResolutionXY;
        const float cascadeDelta = cascadeResReciprocal;
        const size_t numCascades = frame_->csc.cascades.size();

        frame_->csc.worldLightCamera = CameraPtr(new Camera(false, false));
        auto worldLightCamera = frame_->csc.worldLightCamera;
        worldLightCamera->SetAngle(worldLight_->GetRotation());

        // See "Foundations of Game Engine Development, Volume 2: Rendering (pp. 178)
        //
        // FOV_x = 2tan^-1(s/g), FOV_y = 2tan^-1(1/g)
        // ==> tan(FOV_y/2)=1/g ==> g=1/tan(FOV_y/2)
        // where s is the aspect ratio (width / height)

        // Set up the shadow texture offsets
        frame_->csc.cascadeShadowOffsets[0] = glm::vec4(-cascadeDelta, -cascadeDelta, cascadeDelta, -cascadeDelta);
        frame_->csc.cascadeShadowOffsets[1] = glm::vec4(cascadeDelta, cascadeDelta, -cascadeDelta, cascadeDelta);
        // _state.cascadeShadowOffsets[0] = glm::vec4(-cascadeDelta, -cascadeDelta, cascadeDelta, -cascadeDelta);
        // _state.cascadeShadowOffsets[1] = glm::vec4(cascadeDelta, cascadeDelta, -cascadeDelta, cascadeDelta);

        // Assume directional light translation is none
        // Camera light(false);
        // light.setAngle(_state.worldLight.getRotation());
        const Camera & light = *worldLightCamera;
        const Camera & c = *camera_;

        const glm::mat4& lightWorldTransform = light.GetWorldTransform();
        const glm::mat4& lightViewTransform = light.GetViewTransform();
        const glm::mat4& cameraWorldTransform = c.GetWorldTransform();
        const glm::mat4& cameraViewTransform = c.GetViewTransform();
        const glm::mat4 transposeLightWorldTransform = glm::transpose(lightWorldTransform);

        // See page 152, eq. 8.21
        const glm::vec3 worldLightDirWorldSpace = -lightWorldTransform[2];
        const glm::vec3 worldLightDirCamSpace = glm::normalize(glm::mat3(cameraViewTransform) * worldLightDirWorldSpace);
        frame_->csc.worldLightDirectionCameraSpace = worldLightDirCamSpace;

        const glm::mat4 L = lightViewTransform * cameraWorldTransform;

        // @see https://gamedev.stackexchange.com/questions/183499/how-do-i-calculate-the-bounding-box-for-an-ortho-matrix-for-cascaded-shadow-mapp
        // @see https://ogldev.org/www/tutorial49/tutorial49.html
        const float ar = float(Window::Instance()->GetWindowDims().first) / float(Window::Instance()->GetWindowDims().second);
        //const float tanHalfHFov = glm::tan(Radians(_params.fovy).value() / 2.0f) * ar;
        //const float tanHalfVFov = glm::tan(Radians(_params.fovy).value() / 2.0f);
        const float projPlaneDist = glm::tan(Radians(params_.fovy).value() / 2.0f);
        const float znear = 1.0f;//params_.znear; //0.001f; //_params.znear;
        // We don't want zfar to be unbounded, so we constrain it to at most 800 which also has the nice bonus
        // of increasing our shadow map resolution (same shadow texture resolution over a smaller total area)
        const float zfar  = params_.zfar; //std::min(800.0f, _params.zfar);
        frame_->csc.znear = znear;
        frame_->csc.zfar = zfar;

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
            //const float d = floorf(lambda * (log - uniform) + uniform);
            const float d = floorf(lambda * log + (1.0f - lambda) * uniform);
            cascadeEnds[i] = d;
            //STRATUS_LOG << "Cascade " << i << " ends " << d << std::endl;
        }

        // std::vector<float> cascadeEnds = {
        //     5.0f,
        //     20.0f,
        //     100.0f,
        //     200.0f
        // };

        // see https://gamedev.stackexchange.com/questions/183499/how-do-i-calculate-the-bounding-box-for-an-ortho-matrix-for-cascaded-shadow-mapp
        // see https://ogldev.org/www/tutorial49/tutorial49.html
        // We offset each cascade begin from 1 onwards so that there is some overlap between the start of cascade k and the end of cascade k-1
        //const std::vector<float> cascadeBegins = { 0.0f, cascadeEnds[0] - 10.0f,  cascadeEnds[1] - 10.0f, cascadeEnds[2] - 10.0f }; // 4 cascades max
        const std::vector<float> cascadeBegins = { 0.0f, cascadeEnds[0] - 4.0f,  cascadeEnds[1] - 4.0f, cascadeEnds[2] - 4.0f }; // 4 cascades max
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
            frame_->csc.cascades[i].cascadeBegins = ak;
            frame_->csc.cascades[i].cascadeEnds   = bk;
            aks.push_back(ak);
            bks.push_back(bk);

            // These base values are in camera space and define our frustum corners
            const float xn = ak * ar * projPlaneDist;
            const float xf = bk * ar * projPlaneDist;
            const float yn = ak * projPlaneDist;
            const float yf = bk * projPlaneDist;
            // Keep all of these in camera space for now
            std::vector<glm::vec4, Vec4Allocator> frustumCorners({
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
                },

                Vec4Allocator(frame_->perFrameScratchMemory)
            );

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
            const float T = dk / frame_->csc.cascadeResolutionXY;
            frame_->csc.cascades[i].cascadeRadius = dk / 2.0f;

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
            frame_->csc.cascades[i].cascadePositionLightSpace = sk;
            frame_->csc.cascades[i].cascadePositionCameraSpace = glm::vec3(cameraViewTransform * lightWorldTransform * glm::vec4(sk, 1.0f));

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
            // const glm::mat4 cascadeTexelOrthoProjection(glm::vec4(1.0f / dk, 0.0f, 0.0f, 0.0f), 
            //                                            glm::vec4(0.0f, 1.0f / dk, 0.0f, 0.0f),
            //                                            glm::vec4(0.0f, 0.0f, 1.0f / (maxZ - minZ), 0.0f),
            //                                            glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
            const glm::mat4 cascadeTexelOrthoProjection = cascadeOrthoProjection;

            // Note: if we want we can set texelProjection to be cascadeTexelOrthoProjection and then set projectionView
            // to be cascadeTexelOrthoProjection * cascadeViewTransform. This has the added benefit of automatically translating
            // x, y positions to texel coordinates on the range [0, 1] rather than [-1, 1].
            //
            // However, the alternative is to just compute (coordinate * 0.5 + 0.5) in the fragment shader which does the same thing.
            frame_->csc.cascades[i].projectionViewRender = cascadeOrthoProjection * cascadeViewTransform;
            frame_->csc.cascades[i].projectionViewSample = cascadeTexelOrthoProjection * cascadeViewTransform;
            //STRATUS_LOG << _frame->csc.cascades[i].projectionViewSample << std::endl;

            if (i > 0) {
                // See page 187, eq. 8.82
                // Ck = Mk_shadow * (M0_shadow) ^ -1
                glm::mat4 Ck = frame_->csc.cascades[i].projectionViewSample * glm::inverse(frame_->csc.cascades[0].projectionViewSample);
                frame_->csc.cascades[i].sampleCascade0ToCurrent = Ck;

                // This will allow us to calculate the cascade blending weights in the vertex shader and then
                // the cascade indices in the pixel shader
                const glm::vec3 n = -glm::vec3(cameraWorldTransform[2]);
                const glm::vec3 c = glm::vec3(cameraWorldTransform[3]);
                // fk now represents a plane along the direction of the view frustum. Its normal is equal to the camera's forward
                // direction in world space and it contains the point c + ak*n.
                const glm::vec4 fk = glm::vec4(n.x, n.y, n.z, glm::dot(-n, c) - ak) * (1.0f / (bks[i - 1] - ak));
                frame_->csc.cascades[i].cascadePlane = fk;
                //STRATUS_LOG << fk << std::endl;
                //_frame->csc.cascades[i].cascadePlane = glm::vec4(10.0f);
            }
        }
    }

    bool RendererFrontend::EntityChanged_(const EntityPtr& p) {
        auto tc = p->Components().GetComponent<GlobalTransformComponent>().component;
        auto rc = p->Components().GetComponent<RenderComponent>().component;
        return tc->ChangedWithinLastFrame() || rc->ChangedWithinLastFrame();
    }

    void RendererFrontend::CheckEntitySetForChanges_(std::unordered_set<EntityPtr>& set) {
        for (auto& entity : set) {
            // If this is a light-interacting node, run through all the lights to see if they need to be updated
            if (EntityChanged_(entity)) {               

                InitializeMeshTransformComponent(entity);

                frame_->drawCommands->UpdateTransforms(entity);

                if (IsLightInteracting(entity)) {
                    for (const auto& light : lights_) {
                        // Static lights don't care about entity movement changes
                        if (light->IsStaticLight()) continue;

                        auto lightPos = light->GetPosition();
                        auto lightRadius = light->GetRadius();
                        //If the EntityView is in the light's visible set, its shadows are now out of date
                        for (size_t i = 0; i < GetMeshCount(entity); ++i) {
                            if (glm::distance(GetWorldTransform(entity, i), lightPos) > lightRadius) {
                                frame_->lightsToUpdate.PushBack(light);
                            }
                            // If the EntityView has moved inside the light's radius, add it
                            else if (glm::distance(GetWorldTransform(entity, i), lightPos) < lightRadius) {
                                frame_->lightsToUpdate.PushBack(light);
                            }
                        }
                    }
                }
            }
        }
    }

    void RendererFrontend::CheckForEntityChanges_() {
        // We only care about dynamic light-interacting entities
        CheckEntitySetForChanges_(dynamicEntities_);
    }

    void RendererFrontend::MarkDynamicLightsDirty_() {
        for (auto& light : dynamicLights_) {
            frame_->lightsToUpdate.PushBack(light);
        }
    }

    void RendererFrontend::MarkStaticLightsDirty_() {
        for (auto& light : staticLights_) {
            frame_->lightsToUpdate.PushBack(light);
        }
    }

    void RendererFrontend::MarkAllLightsDirty_() {
        for (auto& light : lights_) {
            frame_->lightsToUpdate.PushBack(light);
        }
    }

    void RendererFrontend::UpdateLights_() {
        frame_->lightsToRemove.clear();
        // First get rid of all lights that are pending deletion
        for (auto& light : lightsToRemove_) {
            frame_->lights.erase(light);
            frame_->virtualPointLights.erase(light);
            frame_->lightsToRemove.insert(light);
        }
        lightsToRemove_.clear();

        // Update the world light
        frame_->csc.worldLight = worldLight_;//->Copy();

        // Now go through and update all lights that have changed in some way
        for (auto& light : lights_) {
            if ( !light->CastsShadows() ) continue;

            // See if the light moved or its radius changed
            if (light->PositionChangedWithinLastFrame() || light->RadiusChangedWithinLastFrame()) {
                frame_->lightsToUpdate.PushBack(light);
            }
        }
    }

    void RendererFrontend::UpdateMaterialSet_() {
        frame_->materialInfo->UploadDataToGpu();
    }

    // TODO: This desperately needs to be refactored and made more efficient
    void RendererFrontend::UpdateDrawCommands_() {
        const bool staticLightsDirty = frame_->drawCommands->UploadStaticDataToGpu();
        const bool dynamicLightsDirty = staticLightsDirty || frame_->drawCommands->UploadDynamicDataToGpu();
        frame_->drawCommands->UploadFlatDataToGpu();

        if (staticLightsDirty) MarkStaticLightsDirty_();

        if (dynamicLightsDirty) MarkDynamicLightsDirty_();
    }

    std::vector<glm::vec4, Vec4Allocator> ComputeCornersWithTransform(const GpuAABB& aabb, const glm::mat4& transform, const UnsafePtr<StackAllocator>& perFrameAllocator) {
        glm::vec4 vmin = aabb.vmin.ToVec4();
        glm::vec4 vmax = aabb.vmax.ToVec4();

        std::vector<glm::vec4, Vec4Allocator> corners({
            transform * glm::vec4(vmin.x, vmin.y, vmin.z, 1.0),
            transform * glm::vec4(vmin.x, vmax.y, vmin.z, 1.0),
            transform * glm::vec4(vmin.x, vmin.y, vmax.z, 1.0),
            transform * glm::vec4(vmin.x, vmax.y, vmax.z, 1.0),
            transform * glm::vec4(vmax.x, vmin.y, vmin.z, 1.0),
            transform * glm::vec4(vmax.x, vmax.y, vmin.z, 1.0),
            transform * glm::vec4(vmax.x, vmin.y, vmax.z, 1.0),
            transform * glm::vec4(vmax.x, vmax.y, vmax.z, 1.0)
            },

            Vec4Allocator(perFrameAllocator)
        );

        return corners;
    }

    // This code was taken from "3D Graphics Rendering Cookbook" source code, shared/UtilsMath.h
    GpuAABB TransformAabb(const GpuAABB& aabb, const glm::mat4& transform, const UnsafePtr<StackAllocator>& perFrameAllocator) {
        std::vector<glm::vec4, Vec4Allocator> corners = ComputeCornersWithTransform(aabb, transform, perFrameAllocator);

        glm::vec3 vmin3 = corners[0];
        glm::vec3 vmax3 = corners[0];

        for (int i = 1; i < 8; ++i) {
            vmin3 = glm::min(vmin3, glm::vec3(corners[i]));
            vmax3 = glm::max(vmax3, glm::vec3(corners[i]));
        }

        GpuAABB result;
        result.vmin = glm::vec4(vmin3, 1.0);
        result.vmax = glm::vec4(vmax3, 1.0); 

        return result;
    }

    bool IsAabbVisible(const GpuAABB& aabb, const std::vector<glm::vec4>& frustumPlanes) {
        glm::vec4 vmin = aabb.vmin.ToVec4();
        glm::vec4 vmax = aabb.vmax.ToVec4();

        for (int i = 0; i < 6; ++i) {
            const glm::vec4& g = frustumPlanes[i];
    		if ((glm::dot(g, glm::vec4(vmin.x, vmin.y, vmin.z, 1.0f)) < 0.0) &&
			    (glm::dot(g, glm::vec4(vmax.x, vmin.y, vmin.z, 1.0f)) < 0.0) &&
			    (glm::dot(g, glm::vec4(vmin.x, vmax.y, vmin.z, 1.0f)) < 0.0) &&
			    (glm::dot(g, glm::vec4(vmax.x, vmax.y, vmin.z, 1.0f)) < 0.0) &&
			    (glm::dot(g, glm::vec4(vmin.x, vmin.y, vmax.z, 1.0f)) < 0.0) &&
			    (glm::dot(g, glm::vec4(vmax.x, vmin.y, vmax.z, 1.0f)) < 0.0) &&
			    (glm::dot(g, glm::vec4(vmin.x, vmax.y, vmax.z, 1.0f)) < 0.0) &&
			    (glm::dot(g, glm::vec4(vmax.x, vmax.y, vmax.z, 1.0f)) < 0.0))
            {
                return false;
            }
        }

        return true;
    }

    // bool IsAabbVisible(const GpuAABB& aabb, const std::vector<glm::vec4>& frustumPlanes) {
    //     const glm::vec3 vmin   = aabb.vmin.ToVec4();
    //     const glm::vec3 vmax   = aabb.vmax.ToVec4();
    //     const glm::vec4 center = glm::vec4((vmin + vmax) * 0.5f, 1.0);
    //     const glm::vec4 size   = glm::vec4((vmax - vmin) * 0.5f, 1.0);

    //     for (int i = 0; i < 6; ++i) {
    //         const glm::vec4& g = frustumPlanes[i];
    //         const float rg = std::fabs(g.x * size.x) + std::fabs(g.y * size.y) + std::fabs(g.z * size.z);
    //         if (glm::dot(g, center) <= -rg) {
    //             return false;
    //         }
    //     }

    //     return true;
    // }

    // See the section on culling in "3D Graphics Rendering Cookbook"
    void RendererFrontend::UpdateVisibility_() {   
        using CommandBufferAllocator = StackBasedPoolAllocator< std::unordered_map<RenderFaceCulling, GpuCommandBuffer2Ptr>*>;
        const std::vector<std::unordered_map<RenderFaceCulling, GpuCommandBuffer2Ptr>*, CommandBufferAllocator> commands({
            &frame_->drawCommands->flatMeshes,
            &frame_->drawCommands->dynamicPbrMeshes,
            &frame_->drawCommands->staticPbrMeshes
            },

            CommandBufferAllocator(frame_->perFrameScratchMemory)
        );

        for (auto& buffer : commands) {
            UpdateVisibility_(
                *viscullLodSelect_.get(),
                frame_->projection,
                frame_->camera->GetViewTransform(),
                *buffer,
                true
            );
        }
    }

    void RendererFrontend::UpdateVisibility_(
        Pipeline& pipeline,
        const glm::mat4& projection, 
        const glm::mat4& view, 
        std::unordered_map<RenderFaceCulling, GpuCommandBuffer2Ptr>& inOutDrawCommands,
        const bool selectLods) {

        static const std::vector<RenderFaceCulling> culling{
            RenderFaceCulling::CULLING_CCW,
            RenderFaceCulling::CULLING_CW,
            RenderFaceCulling::CULLING_NONE  
        };

        const glm::mat4 vp = projection * view;
        const glm::mat4 vpt = glm::transpose(vp);
        //const glm::mat4 ivp = glm::inverse(vp);

        // See https://gamedev.stackexchange.com/questions/29999/how-do-i-create-a-bounding-frustum-from-a-view-projection-matrix
        // These corners are in NDC (Normalized Device Coordinate) space which is the space we arrive at after using the view-projection
        // matrix
        // glm::vec4 corners[] = {
		//     glm::vec4(-1, -1, -1, 1), 
        //     glm::vec4( 1, -1, -1, 1),
		//     glm::vec4( 1,  1, -1, 1),  
        //     glm::vec4(-1,  1, -1, 1),
		//     glm::vec4(-1, -1,  1, 1), 
        //     glm::vec4( 1, -1,  1, 1),
		//     glm::vec4( 1,  1,  1, 1),  
        //     glm::vec4(-1,  1,  1, 1)
	    // };

        // // This will convert the corners from NDC -> world space
        // for (int i = 0; i < 8; i++) {
        //     const glm::vec4 q = ivp * corners[i];
        //     corners[i] = q / q.w;
        // }

        std::vector<glm::vec4, Vec4Allocator> frustumPlanes({
            // left, right, bottom, top
            (vpt[3] + vpt[0]),
            (vpt[3] - vpt[0]),
            (vpt[3] + vpt[1]),
            (vpt[3] - vpt[1]),
            // near, far
            (vpt[3] + vpt[2]),
            (vpt[3] - vpt[2]),
            },

            Vec4Allocator(frame_->perFrameScratchMemory)
        );

        pipeline.Bind();

        for (size_t i = 0; i < 6; ++i) {
            pipeline.SetVec4("frustumPlanes[" + std::to_string(i) + "]", frustumPlanes[i]);
        }

        pipeline.SetVec3("viewPosition", frame_->camera->GetPosition());
        pipeline.SetFloat("zfar", frame_->csc.zfar);
        
        for (const auto& cull : culling) {
            auto it = inOutDrawCommands.find(cull);
            if (it->second->NumDrawCommands() == 0) continue;

            it->second->GetIndirectDrawCommandsBuffer(0).BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, 1);
            it->second->GetVisibleDrawCommandsBuffer().BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, 14);
            it->second->BindModelTransformBuffer(2);
            it->second->BindAabbBuffer(3);

            if (selectLods) {
                it->second->GetSelectedLodDrawCommandsBuffer().BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, 13);
                for (size_t k = 0; k < it->second->NumLods(); ++k) {
                    it->second->GetIndirectDrawCommandsBuffer(k).BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, k + 5);
                }
            }

            pipeline.SetUint("numDrawCalls", (unsigned int)(it->second->NumDrawCommands()));
            pipeline.SetMat4("view", view);
            //pipeline.setMat4("view", _frame->camera->getViewTransform());
            //pipeline.setMat4("projection", _frame->projection);
            pipeline.DispatchCompute(1, 1, 1);
            pipeline.SynchronizeCompute();
        }

        pipeline.Unbind();

        //for (size_t i = 0; i < drawCommands.size(); ++i) {
        //   auto& map = drawCommands[i];
        //   for (const auto& cull : culling) {
        //       auto outIt = map->find(cull);
        //       const auto& buffer = outIt->second;
        //       if (buffer->NumDrawCommands() == 0) continue;
        //       int numCommands = 0;

        //       for (size_t k = 0; k < buffer->NumDrawCommands(); ++k) {
        //           GpuAABB aabb = TransformAabb(buffer->aabbs[k], buffer->globalTransforms[k]);
        //           if (!IsAabbVisible(aabb, frustumPlanes)) {
        //               buffer->indirectDrawCommands[k].instanceCount = 0;
        //           }
        //           else {
        //               buffer->indirectDrawCommands[k].instanceCount = 1;
        //               ++numCommands;
        //           }
        //       }

        //       buffer->UploadDataToGpu();

        //       //STRATUS_LOG << "Before/After: " << buffer->NumDrawCommands() << "/" << numCommands << std::endl;
        //   }
        //}
    }

    void RendererFrontend::UpdatePrevFrameModelTransforms_() {
        using CommandBufferAllocator = StackBasedPoolAllocator<std::unordered_map<RenderFaceCulling, GpuCommandBuffer2Ptr>*>;
        std::vector<std::unordered_map<RenderFaceCulling, GpuCommandBuffer2Ptr>*, CommandBufferAllocator> drawCommands({
            &frame_->drawCommands->flatMeshes,
            &frame_->drawCommands->dynamicPbrMeshes,
            &frame_->drawCommands->staticPbrMeshes
            },

            CommandBufferAllocator(frame_->perFrameScratchMemory)
        );

        updateTransforms_->Bind();

        for (const auto& entry : drawCommands) {
            auto ccw = entry->find(RenderFaceCulling::CULLING_CCW);
            auto cw = entry->find(RenderFaceCulling::CULLING_CW);
            auto cnone = entry->find(RenderFaceCulling::CULLING_NONE);

            if (ccw->second->NumDrawCommands() > 0) {
                ccw->second->BindPrevFrameModelTransformBuffer(0);
                ccw->second->BindModelTransformBuffer(1);
            }
            updateTransforms_->SetInt("cull0NumMatrices", ccw->second->NumDrawCommands());

            if (cw->second->NumDrawCommands() > 0) {
                cw->second->BindPrevFrameModelTransformBuffer(2);
                cw->second->BindModelTransformBuffer(3);
            }
            updateTransforms_->SetInt("cull1NumMatrices", cw->second->NumDrawCommands());

            if (cnone->second->NumDrawCommands() > 0) {
                cnone->second->BindPrevFrameModelTransformBuffer(4);
                cnone->second->BindModelTransformBuffer(5);
            }
            updateTransforms_->SetInt("cull2NumMatrices", cnone->second->NumDrawCommands());

            updateTransforms_->DispatchCompute(100, 1, 1);
            updateTransforms_->SynchronizeCompute();
        }

        updateTransforms_->Unbind();
    }
}