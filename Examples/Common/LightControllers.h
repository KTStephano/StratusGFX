#pragma once

#include "LightComponents.h"

struct LightParams {
    glm::vec3 position;
    glm::vec3 color;
    float intensity;
    bool CastsShadows;
    uint32_t numShadowSamples; // only valid for Virtual Point Lights (VPLs)

    LightParams()
        : LightParams(glm::vec3(0.0f), glm::vec3(1.0f), 1.0f) {}

    LightParams(const glm::vec3& position, const glm::vec3& color, const float intensity)
        : LightParams(position, color, intensity, true) {}

    LightParams(const glm::vec3& position, const glm::vec3& color, const float intensity, const bool CastsShadows)
        : LightParams(position, color, intensity, CastsShadows, 3) {}

    LightParams(const glm::vec3& position, const glm::vec3& color, const float intensity, const bool CastsShadows, const uint32_t numShadowSamples)
        : position(position), color(color), intensity(intensity), CastsShadows(CastsShadows), numShadowSamples(numShadowSamples) {}
};

struct LightCreator {
    static void Initialize();
    static void Shutdown();
    static void CreateRandomLightMover(const LightParams&);
    static void CreateStationaryLight(const LightParams&, const bool spawnCube = true);
    static void CreateVirtualPointLight(const LightParams&, const bool spawnCube = false);

private:
    static std::vector<stratus::EntityProcessHandle> handles;
};

struct LightProcess : public stratus::EntityProcess {
    LightProcess();
    virtual ~LightProcess();

    void Process(const double deltaSeconds) override;
    void EntitiesAdded(const std::unordered_set<stratus::EntityPtr>& e) override;
    void EntitiesRemoved(const std::unordered_set<stratus::EntityPtr>& e) override;
    void EntityComponentsAdded(const std::unordered_map<stratus::EntityPtr, std::vector<stratus::EntityComponent*>>& added) override;
    void EntityComponentsEnabledDisabled(const std::unordered_set<stratus::EntityPtr>& changed) override;

    stratus::InputHandlerPtr input;
};

struct RandomLightMoverProcess : public stratus::EntityProcess {
    virtual ~RandomLightMoverProcess() = default;

    void Process(const double deltaSeconds) override;
    void EntitiesAdded(const std::unordered_set<stratus::EntityPtr>& e) override;
    void EntitiesRemoved(const std::unordered_set<stratus::EntityPtr>& e) override;
    void EntityComponentsAdded(const std::unordered_map<stratus::EntityPtr, std::vector<stratus::EntityComponent*>>& added) override;
    void EntityComponentsEnabledDisabled(const std::unordered_set<stratus::EntityPtr>& changed) override;

private:
    static bool IsEntityRelevant_(const stratus::EntityPtr&);
    static void ChangeDirection_(RandomLightMoverComponent *);

private:
    std::unordered_set<stratus::EntityPtr> entities_;
};