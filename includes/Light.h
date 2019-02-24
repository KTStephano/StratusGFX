
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
    float _intensity = 1.0;

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
        r = std::max(0.0f, r);
        g = std::max(0.0f, g);
        b = std::max(0.0f, b);
        _color = glm::vec3(r, g, b);
    }

    /**
     * A light's color values can all be on the range of
     * [0.0, 1.0], but the intensity specifies how strong it
     * should be.
     * @param i
     */
    void setIntensity(float i) {
        if (i < 0) return;
        _intensity = i;
    }

    float getIntensity() const {
        return _intensity;
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
