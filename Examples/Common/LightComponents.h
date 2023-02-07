#pragma once

#include "StratusCommon.h"
#include "glm/glm.hpp"
#include "StratusWindow.h"
#include "StratusRendererFrontend.h"
#include "StratusLog.h"
#include "StratusCamera.h"
#include "StratusLight.h"
#include "StratusEngine.h"
#include "StratusResourceManager.h"
#include "StratusMaterial.h"
#include "StratusUtils.h"
#include "StratusEntityManager.h"
#include "StratusEntityCommon.h"
#include "StratusEntity.h"

ENTITY_COMPONENT_STRUCT(LightComponent)
    stratus::LightPtr light;
    LightComponent(stratus::LightPtr light)
        : light(light) {}

    LightComponent(const LightComponent& other) {
        light = other.light->Copy();
    }
};

ENTITY_COMPONENT_STRUCT(LightCubeComponent)
    stratus::EntityPtr cube;
    LightCubeComponent(stratus::EntityPtr cube)
        : cube(cube) {}

    LightCubeComponent(const LightCubeComponent& other) {
        cube = other.cube->Copy();
    }
};

ENTITY_COMPONENT_STRUCT(RandomLightMoverComponent)
    glm::vec3 position = glm::vec3(0.0f);
    glm::vec3 direction = glm::vec3(0.0f);
    double elapsedSeconds = 0.0;

    RandomLightMoverComponent() {}
    RandomLightMoverComponent(const RandomLightMoverComponent&) = default;
};

