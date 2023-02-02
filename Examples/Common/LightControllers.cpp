#include "LightControllers.h"

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
    light->position = p.position;
}

static void InitCube(const LightParams& p,
                     const stratus::LightPtr& light,
                     stratus::Entity2Ptr& cube) {
    // cube->GetRenderNode()->SetMaterial(INSTANCE(MaterialManager)->CreateDefault());
    // cube->GetRenderNode()->EnableLightInteraction(false);
    // cube->SetLocalScale(glm::vec3(1.0f));
    // cube->SetLocalPosition(p.position);
    // cube->GetRenderNode()->GetMeshContainer(0)->material->SetDiffuseColor(light->getColor());
}

void LightCreator::CreateRandomLightMover(const LightParams& p) {
    auto ptr = stratus::Entity2::Create();
    stratus::LightPtr light(new stratus::PointLight());
    InitLight(p, light);

    stratus::Entity2Ptr cube = INSTANCE(ResourceManager)->CreateCube();
    InitCube(p, light, cube);

    ptr->Components().AttachComponent<LightComponent>(light);
    ptr->Components().AttachComponent<LightCubeComponent>(cube);
    ptr->Components().AttachComponent<RandomLightMoverComponent>();
    auto mover = ptr->Components().GetComponent<RandomLightMoverComponent>().component;
    mover->position = p.position;

    INSTANCE(EntityManager)->AddEntity(ptr);
    INSTANCE(RendererFrontend)->AddLight(light);
    INSTANCE(RendererFrontend)->AddStaticEntity(cube);
}

void LightCreator::CreateStationaryLight(const LightParams& p) {
    auto ptr = stratus::Entity2::Create();
    stratus::LightPtr light(new stratus::PointLight());
    InitLight(p, light);

    stratus::Entity2Ptr cube = INSTANCE(ResourceManager)->CreateCube();
    InitCube(p, light, cube);

    ptr->Components().AttachComponent<LightComponent>(light);
    ptr->Components().AttachComponent<LightCubeComponent>(cube);

    INSTANCE(EntityManager)->AddEntity(ptr);
    INSTANCE(RendererFrontend)->AddLight(light);
    INSTANCE(RendererFrontend)->AddStaticEntity(cube);
}

void LightCreator::CreateVirtualPointLight(const LightParams& p) {
    auto ptr = stratus::Entity2::Create();
    stratus::LightPtr light(new stratus::VirtualPointLight());
    InitLight(p, light);
    ((stratus::VirtualPointLight *)light.get())->SetNumShadowSamples(p.numShadowSamples);

    STRATUS_LOG << "VPL Radius: " << light->getRadius() << std::endl;

    ptr->Components().AttachComponent<LightComponent>(light);
    INSTANCE(EntityManager)->AddEntity(ptr);
    INSTANCE(RendererFrontend)->AddLight(light);
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
                    }
                }
            }
        }
    }

    std::vector<stratus::Entity2Ptr> entitiesToRemove;
    std::vector<stratus::Entity2Ptr> entities;
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

static bool EntityIsRelevant(const stratus::Entity2Ptr& entity) {
    return entity->Components().ContainsComponent<LightComponent>() ||
           entity->Components().ContainsComponent<LightCubeComponent>();
}

void LightProcess::Process(const double deltaSeconds) {
    for (stratus::Entity2Ptr& entity : ConvertHandlerToLightDelete(input)->entitiesToRemove) {
        if (entity->Components().ContainsComponent<LightComponent>()) {
            INSTANCE(RendererFrontend)->RemoveLight(
                entity->Components().GetComponent<LightComponent>().component->light
            );
        }
        
        if (entity->Components().ContainsComponent<LightCubeComponent>()) {
            INSTANCE(RendererFrontend)->RemoveEntity(
                entity->Components().GetComponent<LightCubeComponent>().component->cube
            );
        }
    }
}

void LightProcess::EntitiesAdded(const std::unordered_set<stratus::Entity2Ptr>& e) {
    for (auto entity : e) {
        if ( !EntityIsRelevant(entity) ) continue;
        ConvertHandlerToLightDelete(input)->entities.push_back(entity);
    }
}

void LightProcess::EntitiesRemoved(const std::unordered_set<stratus::Entity2Ptr>& e) {
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

void LightProcess::EntityComponentsAdded(const std::unordered_map<stratus::Entity2Ptr, std::vector<stratus::Entity2Component*>>& added) {
    // Do nothing
}

void LightProcess::EntityComponentsEnabledDisabled(const std::unordered_set<stratus::Entity2Ptr>& changed) {
    // Do nothing
}

void RandomLightMoverProcess::Process(const double deltaSeconds) {
    static const glm::vec3 speed(5.0f);
    for (auto ptr : _entities) {
        LightComponent * light = ptr->Components().GetComponent<LightComponent>().component;
        LightCubeComponent * cube = ptr->Components().GetComponent<LightCubeComponent>().component;
        RandomLightMoverComponent * c = ptr->Components().GetComponent<RandomLightMoverComponent>().component;

        c->position = c->position + speed * c->direction * float(deltaSeconds);
        //cube->cube->SetLocalPosition(c->position);
        light->light->position = c->position;

        c->elapsedSeconds += deltaSeconds;
        if (c->elapsedSeconds > 5.0) {
            c->elapsedSeconds = 0.0;
            _ChangeDirection(c);
        }
    }
}

void RandomLightMoverProcess::EntitiesAdded(const std::unordered_set<stratus::Entity2Ptr>& e) {
    for (auto ptr : e) {
        if (_IsEntityRelevant(ptr)) {
            _entities.insert(ptr);
            _ChangeDirection(ptr->Components().GetComponent<RandomLightMoverComponent>().component);
        }
    }
}

void RandomLightMoverProcess::EntitiesRemoved(const std::unordered_set<stratus::Entity2Ptr>& e) {
    for (auto ptr : e) {
        _entities.erase(ptr);
    }
}

void RandomLightMoverProcess::EntityComponentsAdded(const std::unordered_map<stratus::Entity2Ptr, std::vector<stratus::Entity2Component*>>& added) {

}

void RandomLightMoverProcess::EntityComponentsEnabledDisabled(const std::unordered_set<stratus::Entity2Ptr>& changed) {

}

bool RandomLightMoverProcess::_IsEntityRelevant(const stratus::Entity2Ptr& e) {
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