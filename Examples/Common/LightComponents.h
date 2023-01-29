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
#include "StratusEntity2.h"

ENTITY_COMPONENT_STRUCT(LightComponent)
    stratus::LightPtr light;
    LightComponent(stratus::LightPtr light)
        : light(light) {}
};

ENTITY_COMPONENT_STRUCT(LightCubeComponent)
    stratus::EntityPtr cube;
    LightCubeComponent(stratus::EntityPtr cube)
        : cube(cube) {}
};

ENTITY_COMPONENT_STRUCT(RandomLightMoverComponent)
    glm::vec3 position = glm::vec3(0.0f);
    glm::vec3 speed = glm::vec3(0.0f);
    glm::vec3 direction = glm::vec3(0.0f);
};

