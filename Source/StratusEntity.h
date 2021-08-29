
#ifndef STRATUSGFX_ENTITY_H
#define STRATUSGFX_ENTITY_H

#include "StratusCommon.h"

namespace stratus {
/**
 * This represents an object (separate from a render entity)
 * that can be moved around the world. It does not contain any
 * information related to drawing.
 */
struct Entity {
    glm::vec3 position = glm::vec3(0.0f);
    // Represents: pitch, yaw, roll in degrees
    glm::vec3 rotation = glm::vec3(0.0f);
    // Represents: speedX, speedY, speedZ
    glm::vec3 speed = glm::vec3(0.0f);

    virtual ~Entity() = default;

    void setPosition(float x, float y, float z) {
        position.x = x;
        position.y = y;
        position.z = z;
    }

    void setSpeed(float x, float y, float z) {
        speed.x = x;
        speed.y = y;
        speed.z = z;
    }

    virtual void update(double deltaSeconds) = 0;
};
}

#endif //STRATUSGFX_ENTITY_H
