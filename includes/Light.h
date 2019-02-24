
#ifndef STRATUSGFX_LIGHT_H
#define STRATUSGFX_LIGHT_H

#include "Common.h"

enum class LightType {
    POINTLIGHT,
    SPOTLIGHT,
    DIRECTIONLIGHT
};

class Light {
    glm::vec3 _color = glm::vec3(1.0f);

public:
    glm::vec3 position = glm::vec3(0.0f);
    virtual ~Light() = default;

    /**
     * @return type of point light so that the renderer knows
     *      how to deal with it
     */
    virtual LightType getType() const = 0;

    const glm::vec3 & getColor() const {
        return _color;
    }

    /**
     * Sets the color of the light where the scale
     * is not from [0.0, 1.0] but instead can be any
     * number > 0.0 for each color component. To make this
     * work, HDR support is required.
     */
    void setColor(float r, float g, float b) {
        r = std::max(0.1f, r);
        g = std::max(0.1f, g);
        b = std::max(0.1f, b);
        _color = glm::vec3(r, g, b);
    }
};

class PointLight : public Light {

public:
    ~PointLight() override = default;

    LightType getType() const override {
        return LightType::POINTLIGHT;
    }
};

#endif //STRATUSGFX_LIGHT_H
