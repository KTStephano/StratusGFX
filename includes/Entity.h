
#ifndef STRATUSGFX_ENTITY_H
#define STRATUSGFX_ENTITY_H

#include "Common.h"

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

    virtual void update(double deltaSeconds) = 0;
};

#endif //STRATUSGFX_ENTITY_H
