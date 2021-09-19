#include "StratusRendererFrontend.h"
#include "StratusUtils.h"
#include "StratusLog.h"

namespace stratus {
    RendererFrontend * RendererFrontend::_instance = nullptr;

    RendererFrontend::RendererFrontend(const RendererParams& p)
        : _params(p),
          _frame(std::make_shared<RendererFrame>()) {
        // 4 cascades total
        _frame->csc.cascades.resize(4);
        _frame->csc.cascadeResolutionXY = 4096;
        _frame->csc.regenerateFbo = true;
        _frame->csc.worldLightingEnabled = _worldLight.enabled;

        _renderer = std::make_unique<RendererBackend>(p.viewportWidth, p.viewportHeight, p.appName);
    }

    void RendererFrontend::_AddEntity(const EntityPtr& p, bool& pbrDirty, std::unordered_map<EntityView, EntityStateData>& pbr, std::unordered_map<EntityView, EntityStateData>& flat, std::unordered_map<LightPtr, LightData>& lights) {
        if (p == nullptr || p->GetRenderNode() == nullptr) return;
        EntityStateData state = EntityStateData{p->GetWorldPosition(), p->GetLocalScale(), p->GetLocalRotation().asVec3()};
        if (p->GetRenderNode()->GetLightInteractionEnabled()) {
            const size_t size = pbr.size();
            pbr.insert(std::make_pair(EntityView(p), state));
            pbrDirty = pbrDirty || size != pbr.size();

            for (auto& entry : lights) {
                auto pos = entry.first->position;
                if (glm::distance(p->GetWorldPosition(), pos) < entry.first->getRadius()) {
                    entry.second.visible.insert(EntityView(p));
                    entry.second.dirty = true;
                }
            }
        }
        else {
            flat.insert(std::make_pair(EntityView(p), state));
        }

        for (auto& child : p->GetChildren()) {
            _AddEntity(child, pbrDirty, pbr, flat, lights);
        }
    }

    void RendererFrontend::AddStaticEntity(const EntityPtr& p) {
        auto ul = _LockWrite();
        _AddEntity(p, _staticPbrDirty, _staticPbrEntities, _flatEntities, _lights);
    }

    void RendererFrontend::AddDynamicEntity(const EntityPtr& p) {
        auto ul = _LockWrite();
        _AddEntity(p, _dynamicPbrDirty, _dynamicPbrEntities, _flatEntities, _lights);
    }

    void RendererFrontend::RemoveEntity(const EntityPtr& p) {
        auto ul = _LockWrite();
        auto view = EntityView(p);
        if (_staticPbrEntities.erase(view)) {
            _staticPbrDirty = true;
        }
        else if (_dynamicPbrEntities.erase(view)) {
            _dynamicPbrDirty = true;
        }
        else {
            _flatEntities.erase(view);
        }

        for (auto& entry : _lights) {
            if (entry.second.visible.erase(view)) {
                entry.second.dirty = true;
            }
        }

        for (auto child : p->GetChildren()) RemoveEntity(child);
    }

    void RendererFrontend::ClearEntities() {
        auto ul = _LockWrite();
        _staticPbrEntities.clear();
        _dynamicPbrEntities.clear();
        _flatEntities.clear();

        for (auto& entry : _lights) {
            entry.second.visible.clear();
            entry.second.dirty = true;
        }

        _staticPbrDirty = true;
        _dynamicPbrDirty = true;
    }

    void RendererFrontend::_AttemptAddEntitiesForLight(const LightPtr& light, LightData& data, const std::unordered_map<EntityView, EntityStateData>& entities) {
        auto pos = light->position;
        for (auto& e : entities) {
            if (glm::distance(pos, e.first.Get()->GetWorldPosition()) < light->getRadius()) {
                data.visible.insert(e.first);
                data.dirty = true;
            }
        }
    }

    void RendererFrontend::AddLight(const LightPtr& light) {
        auto ul = _LockWrite();
        if (_lights.find(light) != _lights.end()) return;

        _lights.insert(std::make_pair(light, LightData()));
        _lightsDirty = true;

        auto& data = _lights.find(light)->second;
        data.dirty = true;
        data.lightCopy = light->Copy();

        if ( !light->castsShadows() ) return;

        _AttemptAddEntitiesForLight(light, data, _staticPbrEntities);
        _AttemptAddEntitiesForLight(light, data, _dynamicPbrEntities);
    }

    void RendererFrontend::RemoveLight(const LightPtr& light) {
        auto ul = _LockWrite();
        if (_lights.find(light) == _lights.end()) return;
        auto copy = _lights.find(light)->second.lightCopy;
        _lights.erase(light);
        _lightsToRemove.insert(copy);
        _lightsDirty = true;
    }

    void RendererFrontend::ClearLights() {
        auto ul = _LockWrite();
        for (auto& light : _lights) {
            _lightsToRemove.insert(light.second.lightCopy);
        }
        _lights.clear();
        _lightsDirty = true;
    }

    void RendererFrontend::SetWorldLightingEnabled(const bool enabled) {
        auto ul = _LockWrite();
        _worldLight.enabled = enabled;
    }

    void RendererFrontend::SetWorldLightColor(const glm::vec3& color) {
        auto ul = _LockWrite();
        _worldLight.color = color;
    }

    void RendererFrontend::SetWorldLightIntensity(float intensity) {
        if (intensity < 0.0f) return;
        auto ul = _LockWrite();
        _worldLight.intensity = intensity;
    }

    void RendererFrontend::SetWorldLightRotation(const Rotation& rot) {
        auto ul = _LockWrite();
        _worldLight.rotation = rot;
    }

    void RendererFrontend::SetCamera(const CameraPtr& camera) {
        auto ul = _LockWrite();
        _camera = camera;
    }

    void RendererFrontend::SetViewportDims(const uint32_t width, const uint32_t height) {
        auto ul = _LockWrite();
        _params.viewportWidth = width;
        _params.viewportHeight = height;
        _viewportDirty = true;
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

    std::vector<SDL_Event> RendererFrontend::PollInputEvents() {
        auto ul = _LockWrite();
        return std::move(_events);
    }

    void RendererFrontend::Update(const double deltaSeconds) {
        auto ul = _LockWrite();
        if (_camera == nullptr) return;

        _frame->camera = _camera->Copy();

        _UpdateViewport();
        _UpdateCascadeTransforms();
        _CheckForEntityChanges();
        _UpdateLights();
        _UpdateCameraVisibility();
        _UpdateCascadeVisibility();

        // Update events
        for (auto e : _renderer->PollInputEvents()) {
            _events.push_back(e);
        }

        // Render the scene
        _renderer->Begin(_frame, true);

        // Before rendering, process tasks
        for (auto& task : _rendererTasks) task();
        _rendererTasks.clear();

        _renderer->RenderScene();
        _renderer->End();

        // Clear all light dirty flags
        for (auto& entry : _lights) {
            entry.second.dirty = false;
        }

        // This needs to be unset
        _frame->csc.regenerateFbo = false;
    }

    void RendererFrontend::QueueRendererThreadTask(const Thread::ThreadFunction& task) {
        auto ul = _LockWrite();
        _rendererTasks.push_back(task);
    }

    void RendererFrontend::_UpdateViewport() {
        if (_viewportDirty) {
            const float aspect = _params.viewportWidth / float(_params.viewportHeight);
            _projection        = glm::perspective(
                Radians(_params.fovy).value(),
                aspect,
                _params.znear,
                _params.zfar
            );

            _frame->viewportDirty  = true;
            _frame->projection     = _projection;
            _frame->viewportWidth  = _params.viewportWidth;
            _frame->viewportHeight = _params.viewportHeight;

            _viewportDirty = false;
        }
        else {
            _frame->viewportDirty = false;
        }
    }

    void RendererFrontend::_UpdateCascadeTransforms() {
        const float cascadeResReciprocal = 1.0f / _frame->csc.cascadeResolutionXY;
        const float cascadeDelta = cascadeResReciprocal;
        const size_t numCascades = _frame->csc.cascades.size();

        _frame->csc.worldLightCamera = CameraPtr(new Camera(false));
        auto worldLightCamera = _frame->csc.worldLightCamera;
        worldLightCamera->setAngle(_worldLight.rotation);

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
        const glm::mat4 transposeLightWorldTransform = glm::transpose(lightWorldTransform);

        const glm::mat4 L = lightViewTransform * cameraWorldTransform;

        const float s = float(_params.viewportWidth) / float(_params.viewportHeight);
        // g is also known as the camera's focal length
        const float g = 1.0f / tangent(_params.fovy / 2.0f).value();
        const float znear = _params.znear;
        // We don't want zfar to be unbounded, so we constrain it to at most 600 which also has the nice bonus
        // of increasing our shadow map resolution (same shadow texture resolution over a smaller total area)
        const float zfar  = std::min(600.0f, _params.zfar);

        // @see https://johanmedestrom.wordpress.com/2016/03/18/opengl-cascaded-shadow-maps/
        // @see https://johanmedestrom.wordpress.com/2016/03/18/opengl-cascaded-shadow-maps/
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
            const float d = std::floorf(lambda * (log - uniform) + uniform);
            cascadeEnds[i] = d;
        }

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
            const float baseAkX = (ak * s) / g;
            const float baseAkY = ak / g;
            const float baseBkX = (bk * s) / g;
            const float baseBkY = bk / g;
            // Keep all of these in camera space for now
            std::vector<glm::vec4> frustumCorners = {
                // Near corners
                glm::vec4(baseAkX, -baseAkY, -ak, 1.0f),
                glm::vec4(baseAkX, baseAkY, -ak, 1.0f),
                glm::vec4(-baseAkX, baseAkY, -ak, 1.0f),
                glm::vec4(-baseAkX, -baseAkY, -ak, 1.0f),

                // Far corners
                glm::vec4(baseBkX, -baseBkY, -bk, 1.0f),
                glm::vec4(baseBkX, baseBkY, -bk, 1.0f),
                glm::vec4(-baseBkX, baseBkY, -bk, 1.0f),
                glm::vec4(-baseBkX, -baseBkY, -bk, 1.0f),
            };
            
            // This tells us the maximum diameter for the cascade bounding box
            const float dk = std::ceilf(std::max<float>(glm::length(frustumCorners[0] - frustumCorners[6]), 
                                                        glm::length(frustumCorners[4] - frustumCorners[6])));
            dks.push_back(dk);
            // T is essentially the physical width/height of area corresponding to each texel in the shadow map
            const float T = dk / _frame->csc.cascadeResolutionXY;
            _frame->csc.cascades[i].cascadeRadius = dk / 2.0f;

            // Compute min/max of each so that we can combine it with dk to create a perfectly rectangular bounding box
            float minX = std::numeric_limits<float>::max();
            float maxX = std::numeric_limits<float>::min();
            float minY = minX;
            float maxY = maxX;
            float minZ = minX;
            float maxZ = maxX;
            for (int j = 0; j < frustumCorners.size(); ++j) {
                // First make sure to transform frustumCorners[j] from camera space to light space
                frustumCorners[j] = L * frustumCorners[j];

                minX = std::min(minX, frustumCorners[j].x);
                maxX = std::max(maxX, frustumCorners[j].x);

                minY = std::min(minY, frustumCorners[j].y);
                maxY = std::max(maxY, frustumCorners[j].y);

                minZ = std::min(minZ, frustumCorners[j].z);
                maxZ = std::max(maxZ, frustumCorners[j].z);
            }

            zmins.push_back(minZ);
            zmaxs.push_back(maxZ);

            // Now we calculate cascade camera position sk using the min, max, dk and T for a stable location
            glm::vec3 sk(std::floorf((maxX + minX) / (2.0f * T)) * T, 
                        std::floorf((maxY + minY) / (2.0f * T)) * T, 
                        minZ);
            //sk = glm::vec3(L * glm::vec4(sk, 1.0f));
            // STRATUS_LOG << "sk " << sk << std::endl;
            sks.push_back(sk);
            _frame->csc.cascades[i].cascadePosition = sk;

            // We use transposeLightWorldTransform because it's less precision-error-prone than just doing glm::inverse(lightWorldTransform)
            // Note: we use -sk instead of lightWorldTransform * sk because we're assuming the translation component is 0
            const glm::mat4 cascadeViewTransform = glm::mat4(transposeLightWorldTransform[0], 
                                                            transposeLightWorldTransform[1],
                                                            transposeLightWorldTransform[2],
                                                            glm::vec4(-sk, 1.0f));

            // We add this into the cascadeOrthoProjection map to add a slight depth offset to each value which helps reduce flickering artifacts
            const float shadowDepthOffset = 2e-19;
            // We are putting the light camera location sk on the near plane in the halfway point between left, right, top and bottom planes
            // so it enables us to use the simplified Orthographic Projection matrix below
            //
            // This results in values between [-1, 1]
            const glm::mat4 cascadeOrthoProjection(glm::vec4(2.0f / dk, 0.0f, 0.0f, 0.0f), 
                                                   glm::vec4(0.0f, 2.0f / dk, 0.0f, 0.0f),
                                                   glm::vec4(0.0f, 0.0f, 1.0f / (maxZ - minZ), shadowDepthOffset),
                                                   glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));

            // Gives us x, y values between [0, 1]
            const glm::mat4 cascadeTexelOrthoProjection(glm::vec4(1.0f / dk, 0.0f, 0.0f, 0.0f), 
                                                        glm::vec4(0.0f, 1.0f / dk, 0.0f, 0.0f),
                                                        glm::vec4(0.0f, 0.0f, 1.0f / (maxZ - minZ), 0.0f),
                                                        glm::vec4(0.5f, 0.5f, 0.0f, 1.0f));

            // Note: if we want we can set texelProjection to be cascadeTexelOrthoProjection and then set projectionView
            // to be cascadeTexelOrthoProjection * cascadeViewTransform. This has the added benefit of automatically translating
            // x, y positions to texel coordinates on the range [0, 1] rather than [-1, 1].
            //
            // However, the alternative is to just compute (coordinate * 0.5 + 0.5) in the fragment shader which does the same thing.
            _frame->csc.cascades[i].projectionViewRender = cascadeOrthoProjection * cascadeViewTransform;
            _frame->csc.cascades[i].projectionViewSample = cascadeTexelOrthoProjection * cascadeViewTransform;
            //STRATUS_LOG << _frame->csc.cascades[i].projectionViewSample << std::endl;

            if (i > 0) {
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

    bool RendererFrontend::_EntityChanged(const EntityView& view, const EntityStateData& data) {
        auto position = view.Get()->GetWorldPosition();
        auto scale = view.Get()->GetLocalScale();
        auto rotation = view.Get()->GetLocalRotation().asVec3();

        return glm::distance(position, data.lastPosition) > 0.01f ||
               glm::distance(scale, data.lastScale)       > 0.01f ||
               glm::distance(rotation, data.lastRotation) > 0.01f;
    }

    void RendererFrontend::_CheckEntitySetForChanges(std::unordered_map<EntityView, EntityStateData>& map, bool& flag) {
        for (auto& entry : map) {
            EntityView view = entry.first;
            EntityStateData& data = entry.second;
            if (_EntityChanged(view, data)) {                 
                // Update the cached info
                data.lastPosition = view.Get()->GetWorldPosition();
                data.lastScale = view.Get()->GetLocalScale();
                data.lastRotation = view.Get()->GetLocalRotation().asVec3();
                flag = true;

                RenderNodePtr rnode = view.Get()->GetRenderNode();
                // If this is a light-interacting node, run through all the lights to see if they need to be updated
                if (rnode->GetLightInteractionEnabled()) {
                    for (auto entry : _lights) {
                        auto lightPos = entry.first->position;
                        auto lightRadius = entry.first->getRadius();
                        // If the EntityView is in the light's visible set, its shadows are now out of date
                        if (entry.second.visible.find(view) != entry.second.visible.end()) {
                            // If the EntityView has moved out of the light radius, remove it
                            if (glm::distance(data.lastPosition, lightPos) > lightRadius) {
                                entry.second.visible.erase(view);
                            }
                            entry.second.dirty = true;
                        }
                        // If the EntityView has moved inside the light's radius, add it
                        else if (glm::distance(data.lastPosition, lightPos) < lightRadius) {
                            entry.second.visible.insert(view);
                            entry.second.dirty = true;
                        }
                    }
                }
            }
        }
    }

    void RendererFrontend::_CheckForEntityChanges() {
        std::vector<EntityView> pbrToRemove;
        std::vector<EntityView> flatToRemove;

        // We only care about dynamic light-interacting entities
        _CheckEntitySetForChanges(_dynamicPbrEntities, _dynamicPbrDirty);
    }

    static void UpdateInstancedData(const std::unordered_set<EntityView>& entities, InstancedData& instanced) {
        for (auto& e : entities) {
            auto view = RenderNodeView(e.Get()->GetRenderNode()->Copy());
            if (instanced.find(view) == instanced.end()) {
                std::vector<RendererEntityData> instanceData(view.Get()->GetNumMeshContainers());
                instanced.insert(std::make_pair(view, instanceData));
            }

            auto& entityDataVec = instanced.find(view)->second;
            // Each mesh will have its own instanced data
            for (int i = 0; i < view.Get()->GetNumMeshContainers(); ++i) {
                auto& entityData = entityDataVec[i];
                auto  meshData   = view.Get()->GetMeshContainer(i);
                entityData.dirty = true;
                entityData.modelMatrices.push_back(e.Get()->GetWorldTransform());
                entityData.diffuseColors.push_back(meshData->material->GetDiffuseColor());
                entityData.baseReflectivity.push_back(meshData->material->GetBaseReflectivity());
                entityData.roughness.push_back(meshData->material->GetRoughness());
                entityData.metallic.push_back(meshData->material->GetMetallic());
                ++entityData.size;
            }
        }
    }

    void RendererFrontend::_UpdateLights() {
        // First get rid of all lights that are pending deletion
        for (auto& light : _lightsToRemove) {
            _frame->lights.erase(light);
        }
        _lightsToRemove.clear();

        // Now go through and update all lights that have changed in some way
        for (auto& entry : _lights) {
            auto  light = entry.first;
            auto& data  = entry.second;
            auto  lightCopy = entry.second.lightCopy;

            // See if the light moved or its radius changed
            auto prevPos = data.lightCopy->position;
            auto prevRadius = data.lightCopy->getRadius();
            if (glm::distance(prevPos, light->position) > 0.01f || std::fabs(light->getRadius() - prevRadius) > 0.01f) {
                *data.lightCopy = *light;
                data.dirty = true;
                data.visible.clear();
                if (light->castsShadows()) {
                    _AttemptAddEntitiesForLight(light, data, _staticPbrEntities);
                    _AttemptAddEntitiesForLight(light, data, _dynamicPbrEntities);
                }
            }

            // Rebuild the instance data if necessary
            if (_frame->lights.find(lightCopy) == _frame->lights.end() || data.dirty == true) {
                _frame->lights.insert(std::make_pair(lightCopy, RendererLightData()));
                if ( !lightCopy->castsShadows() ) continue;

                auto& lightData = _frame->lights.find(lightCopy)->second;
                lightData.dirty = true;
                UpdateInstancedData(data.visible, lightData.visible);
            }
            else {
                _frame->lights.find(lightCopy)->second.dirty = data.dirty;
            }
        }
    }

    void RendererFrontend::_UpdateCameraVisibility() {
        const auto pbrEntitySets = std::vector<const std::unordered_map<EntityView, EntityStateData> *>{
            &_staticPbrEntities,
            &_dynamicPbrEntities
        };

        const auto flatEntitySets = std::vector<const std::unordered_map<EntityView, EntityStateData> *>{
            &_flatEntities
        };

        std::unordered_set<EntityView> visiblePbr;
        std::unordered_set<EntityView> visibleFlat;

        _frame->instancedPbrMeshes.clear();
        _frame->instancedFlatMeshes.clear();
        auto position = _camera->getPosition();

        for (const std::unordered_map<EntityView, EntityStateData> * entities : pbrEntitySets) {
            for (auto& entityView : *entities) {
                if (glm::distance(position, entityView.first.Get()->GetWorldPosition()) < _params.zfar) {
                    visiblePbr.insert(entityView.first);
                }
            }
        }

        for (const std::unordered_map<EntityView, EntityStateData> * entities : flatEntitySets) {
            for (auto& entityView : *entities) {
                if (glm::distance(position, entityView.first.Get()->GetWorldPosition()) < _params.zfar) {
                    visibleFlat.insert(entityView.first);
                }
            }
        }
        
        UpdateInstancedData(visiblePbr, _frame->instancedPbrMeshes);
        UpdateInstancedData(visibleFlat, _frame->instancedFlatMeshes);
    }

    void RendererFrontend::_UpdateCascadeVisibility() {
        const auto pbrEntitySets = std::vector<const std::unordered_map<EntityView, EntityStateData> *>{
            &_staticPbrEntities,
            &_dynamicPbrEntities
        };

        std::vector<std::unordered_set<EntityView>> visible(_frame->csc.cascades.size());

        _frame->csc.worldLightingEnabled = _worldLight.enabled;
        _frame->csc.worldLightColor = _worldLight.color * _worldLight.intensity;
        
        for (const std::unordered_map<EntityView, EntityStateData> * entities : pbrEntitySets) {
            for (auto& entityView : *entities) {
                for (int i = 0; i < _frame->csc.cascades.size(); ++i) {
                    // Use diameter as radius so that even entities slightly out of view can contribute
                    // to the shadows (z clamping enabled)
                    const float radius = _frame->csc.cascades[i].cascadeRadius * 2.0f;
                    glm::vec3 position = _frame->csc.cascades[i].cascadePosition;
                    //position = position + position * (radius / 2.0f); // Position in center

                    //if (glm::distance(position, entityView.first.Get()->GetWorldPosition()) < radius) {
                    if (true) {
                        visible[i].insert(entityView.first);
                    }
                }
            }
        }

        for (int i = 0; i < _frame->csc.cascades.size(); ++i) {
            _frame->csc.cascades[i].visible.clear();
            UpdateInstancedData(visible[i], _frame->csc.cascades[i].visible);
        }
    }
}