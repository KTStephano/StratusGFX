#include "LightControllers.h"
#include "StratusRenderComponents.h"

std::vector<stratus::EntityProcessHandle> LightCreator::handles;

void LightCreator::Initialize() {
    Shutdown();

    handles.push_back(INSTANCE(EntityManager)->RegisterEntityProcess<LightProcess>());
    handles.push_back(INSTANCE(EntityManager)->RegisterEntityProcess<RandomLightMoverProcess>());
}

void LightCreator::Shutdown() {
    for (auto handle : handles) {
        INSTANCE(EntityManager)->UnregisterEntityProcess(handle);
    }

    handles.clear();
}

static void InitLight(const LightParams& p, stratus::LightPtr& light) {
    light->setIntensity(p.intensity);
    light->setColor(p.color);
    light->SetPosition(p.position);
    light->setCastsShadows(p.castsShadows);
}

static void InitCube(const LightParams& p,
                     const stratus::LightPtr& light,
                     stratus::EntityPtr& cube) {
    // cube->GetRenderNode()->SetMaterial(INSTANCE(MaterialManager)->CreateDefault());
    // cube->GetRenderNode()->EnableLightInteraction(false);
    // cube->SetLocalScale(glm::vec3(1.0f));
    // cube->SetLocalPosition(p.position);
    // cube->GetRenderNode()->GetMeshContainer(0)->material->SetDiffuseColor(light->getColor());
    auto rc = cube->Components().GetComponent<stratus::RenderComponent>().component;
    auto local = cube->Components().GetComponent<stratus::LocalTransformComponent>().component;
    rc->SetMaterialAt(INSTANCE(MaterialManager)->CreateDefault(), 0);
    cube->Components().DisableComponent<stratus::LightInteractionComponent>();
    local->SetLocalScale(glm::vec3(0.25f));
    local->SetLocalPosition(p.position);
    auto color = light->getColor();
    // This prevents the cube from being so bright that the bloom post fx causes it to glow
    // to an extreme amount
    color = (color / stratus::maxLightColor) * 100.0f;
    rc->GetMaterialAt(0)->SetDiffuseColor(glm::vec4(color, 1.0f));
}

void LightCreator::CreateRandomLightMover(const LightParams& p) {
    auto ptr = stratus::Entity::Create();
    stratus::LightPtr light(new stratus::PointLight(/* staticLight = */ false));
    InitLight(p, light);

    stratus::EntityPtr cube = INSTANCE(ResourceManager)->CreateCube();
    InitCube(p, light, cube);
    cube->Components().DisableComponent<stratus::StaticObjectComponent>();

    ptr->Components().AttachComponent<LightComponent>(light);
    ptr->Components().AttachComponent<LightCubeComponent>(cube);
    ptr->Components().AttachComponent<RandomLightMoverComponent>();
    auto mover = ptr->Components().GetComponent<RandomLightMoverComponent>().component;
    mover->position = p.position;

    INSTANCE(EntityManager)->AddEntity(ptr);
    INSTANCE(RendererFrontend)->AddLight(light);
    INSTANCE(EntityManager)->AddEntity(cube);
    //INSTANCE(RendererFrontend)->AddDynamicEntity(cube);
}

void LightCreator::CreateStationaryLight(const LightParams& p, const bool spawnCube) {
    auto ptr = stratus::Entity::Create();
    stratus::LightPtr light(new stratus::PointLight(/* staticLight = */ false));
    InitLight(p, light);

    stratus::EntityPtr cube;
    if (spawnCube) {
        cube = INSTANCE(ResourceManager)->CreateCube();
        InitCube(p, light, cube);
    }

    ptr->Components().AttachComponent<LightComponent>(light);
    if (spawnCube) ptr->Components().AttachComponent<LightCubeComponent>(cube);

    INSTANCE(EntityManager)->AddEntity(ptr);
    INSTANCE(RendererFrontend)->AddLight(light);
    if (spawnCube) INSTANCE(EntityManager)->AddEntity(cube);
    //INSTANCE(RendererFrontend)->AddDynamicEntity(cube);
}

void LightCreator::CreateVirtualPointLight(const LightParams& p, const bool spawnCube) {
    auto ptr = stratus::Entity::Create();
    stratus::LightPtr light(new stratus::VirtualPointLight());
    InitLight(p, light);
    ((stratus::VirtualPointLight *)light.get())->SetNumShadowSamples(p.numShadowSamples);

    stratus::EntityPtr cube;
    if (spawnCube) {
        cube = INSTANCE(ResourceManager)->CreateCube();
        InitCube(p, light, cube);
        cube->Components().DisableComponent<stratus::StaticObjectComponent>();
    }

    STRATUS_LOG << "VPL Radius: " << light->getRadius() << std::endl;

    ptr->Components().AttachComponent<LightComponent>(light);
    if (spawnCube) ptr->Components().AttachComponent<LightCubeComponent>(cube);
    INSTANCE(EntityManager)->AddEntity(ptr);
    INSTANCE(RendererFrontend)->AddLight(light);
    if (spawnCube) INSTANCE(EntityManager)->AddEntity(cube);
}

struct LightDeleteController : public stratus::InputHandler {
    LightDeleteController() {
    }

    virtual ~LightDeleteController() {
    }

    void HandleInput(const stratus::MouseState& mouse, const std::vector<SDL_Event>& input, const double deltaSeconds) {
        for (auto e : input) {
            switch (e.type) {
                case SDL_KEYDOWN:
                case SDL_KEYUP: {
                    bool released = e.type == SDL_KEYUP;
                    SDL_Scancode key = e.key.keysym.scancode;
                    switch (key) {
                        case SDL_SCANCODE_C: {
                            for (auto& light : entities) {
                                entitiesToRemove.push_back(light);
                            }
                            entities.clear();
                            break;
                        }
                        case SDL_SCANCODE_Z: {
                            if (released) {
                                if (entities.size() > 0) {
                                    entitiesToRemove.push_back(entities[entities.size() - 1]);
                                    entities.pop_back();
                                }
                            }
                            break;
                        }
                        case SDL_SCANCODE_L: {
                            printLights = true;
                            break;
                        }
                    }
                }
            }
        }
    }

    std::vector<stratus::EntityPtr> entitiesToRemove;
    std::vector<stratus::EntityPtr> entities;
    bool printLights = false;
};

LightDeleteController * ConvertHandlerToLightDelete(const stratus::InputHandlerPtr& input) {
    return dynamic_cast<LightDeleteController *>(input.get());
}

LightProcess::LightProcess() {
    input = stratus::InputHandlerPtr(new LightDeleteController());
    INSTANCE(InputManager)->AddInputHandler(input);
}

LightProcess::~LightProcess() {
    auto manager = INSTANCE(InputManager);
    if (manager) {
        manager->RemoveInputHandler(input);
    }
}

static bool EntityIsRelevant(const stratus::EntityPtr& entity) {
    return entity->Components().ContainsComponent<LightComponent>() ||
           entity->Components().ContainsComponent<LightCubeComponent>();
}

void LightProcess::Process(const double deltaSeconds) {
    if (ConvertHandlerToLightDelete(input)->printLights) {
        ConvertHandlerToLightDelete(input)->printLights = false;
        const auto& lights = ConvertHandlerToLightDelete(input)->entities;
        for (const auto& light : lights) {
            auto ptr = stratus::GetComponent<LightComponent>(light)->light;
            const bool containsCube = stratus::ContainsComponent<LightCubeComponent>(light);
            STRATUS_LOG << "LightCreator::CreateStationaryLight(\n"
                        << "    LightParams(glm::vec3" << ptr->GetPosition() << ", "
                        << "glm::vec3" << ptr->getBaseColor() << ", "
                        << ptr->getIntensity() << ", "
                        << (ptr->castsShadows() ? "true" : "false") << "), \n"
                        << "    " << (containsCube ? "true" : "false") << "\n);";
        }
    }

    for (stratus::EntityPtr& entity : ConvertHandlerToLightDelete(input)->entitiesToRemove) {
        if (entity->Components().ContainsComponent<LightComponent>()) {
            INSTANCE(RendererFrontend)->RemoveLight(
                entity->Components().GetComponent<LightComponent>().component->light
            );
        }
        
        if (entity->Components().ContainsComponent<LightCubeComponent>()) {
            INSTANCE(EntityManager)->RemoveEntity(
                entity->Components().GetComponent<LightCubeComponent>().component->cube
            );
            //INSTANCE(RendererFrontend)->RemoveEntity(
            //    entity->Components().GetComponent<LightCubeComponent>().component->cube
            //);
        }
    }
}

void LightProcess::EntitiesAdded(const std::unordered_set<stratus::EntityPtr>& e) {
    for (auto entity : e) {
        if ( !EntityIsRelevant(entity) ) continue;
        ConvertHandlerToLightDelete(input)->entities.push_back(entity);
    }
}

void LightProcess::EntitiesRemoved(const std::unordered_set<stratus::EntityPtr>& e) {
    for (auto entity : e) {
        if ( !EntityIsRelevant(entity) ) continue;
        auto lightDelete = ConvertHandlerToLightDelete(input);
        for (auto it = lightDelete->entities.begin(); it != lightDelete->entities.end(); ++it) {
            if (*it == entity) {
                lightDelete->entities.erase(it);
            }
        }
    }
}

void LightProcess::EntityComponentsAdded(const std::unordered_map<stratus::EntityPtr, std::vector<stratus::EntityComponent*>>& added) {
    // Do nothing
}

void LightProcess::EntityComponentsEnabledDisabled(const std::unordered_set<stratus::EntityPtr>& changed) {
    // Do nothing
}

void RandomLightMoverProcess::Process(const double deltaSeconds) {
    static const glm::vec3 speed(5.0f);
    for (auto ptr : _entities) {
        LightComponent * light = ptr->Components().GetComponent<LightComponent>().component;
        LightCubeComponent * cube = ptr->Components().GetComponent<LightCubeComponent>().component;
        RandomLightMoverComponent * c = ptr->Components().GetComponent<RandomLightMoverComponent>().component;

        c->position = c->position + speed * c->direction * float(deltaSeconds);
        auto cubeTransform = stratus::GetComponent<stratus::LocalTransformComponent>(cube->cube);
        cubeTransform->SetLocalPosition(c->position);
        //cube->cube->SetLocalPosition(c->position);
        light->light->SetPosition(c->position);

        c->elapsedSeconds += deltaSeconds;
        if (c->elapsedSeconds > 5.0) {
            c->elapsedSeconds = 0.0;
            _ChangeDirection(c);
        }
    }
}

void RandomLightMoverProcess::EntitiesAdded(const std::unordered_set<stratus::EntityPtr>& e) {
    for (auto ptr : e) {
        if (_IsEntityRelevant(ptr)) {
            _entities.insert(ptr);
            _ChangeDirection(ptr->Components().GetComponent<RandomLightMoverComponent>().component);
        }
    }
}

void RandomLightMoverProcess::EntitiesRemoved(const std::unordered_set<stratus::EntityPtr>& e) {
    for (auto ptr : e) {
        _entities.erase(ptr);
    }
}

void RandomLightMoverProcess::EntityComponentsAdded(const std::unordered_map<stratus::EntityPtr, std::vector<stratus::EntityComponent*>>& added) {

}

void RandomLightMoverProcess::EntityComponentsEnabledDisabled(const std::unordered_set<stratus::EntityPtr>& changed) {

}

bool RandomLightMoverProcess::_IsEntityRelevant(const stratus::EntityPtr& e) {
    return e->Components().ContainsComponent<RandomLightMoverComponent>() &&
        e->Components().ContainsComponent<LightComponent>() &&
        e->Components().ContainsComponent<LightCubeComponent>();
}

void RandomLightMoverProcess::_ChangeDirection(RandomLightMoverComponent * c) {
        float xModifier = rand() % 100 > 50 ? -1.0f : 1.0f;
        float yModifier = 0.0; // rand() % 100 > 50 ? -1.0f : 1.0f;
        float zModifier = rand() % 100 > 50 ? -1.0f : 1.0f;
        c->direction.x = (rand() % 100) > 50 ? 1.0f : 0.0f;
        c->direction.y = (rand() % 100) > 50 ? 1.0f : 0.0f;
        c->direction.z = (rand() % 100) > 50 ? 1.0f : 0.0f;

        c->direction = c->direction * glm::vec3(xModifier, yModifier, zModifier);
}