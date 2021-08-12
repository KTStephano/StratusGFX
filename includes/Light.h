
#ifndef STRATUSGFX_LIGHT_H
#define STRATUSGFX_LIGHT_H

#include "Common.h"
#include <algorithm>
#include <cmath>

namespace stratus {
enum class LightType {
    POINTLIGHT,
    SPOTLIGHT,
    DIRECTIONLIGHT
};

class Light {
    glm::vec3 _color = glm::vec3(1.0f);
    float _intensity = 1.0f;
    float _radius = 1.0f;

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
        _recalcRadius();
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
        _recalcRadius();
    }

    float getIntensity() const {
        return _intensity;
    }

    float getRadius() const {
        return _radius;
    }

private:
    // See https://learnopengl.com/Advanced-Lighting/Deferred-Shading for the equation
    void _recalcRadius() {
        static const float lightMin = 256.0 / 5;
        const glm::vec3 intensity = getIntensity() * getColor();
        const float Imax = std::max(intensity.x, std::max(intensity.y, intensity.z));
        this->_radius = 2 * std::sqrtf(-4 * (1.0 - Imax * lightMin)) / 2;
    }
};

class PointLight : public Light {
    friend class Renderer;
    
    ShadowMapHandle _shadowHap = -1;

    // These are used to set up the light view matrix
    float lightNearPlane = 0.25f;
    float lightFarPlane = 500.0f;

public:
    ~PointLight() override = default;

    LightType getType() const override {
        return LightType::POINTLIGHT;
    }

    ShadowMapHandle getShadowMapHandle() const {
        return this->_shadowHap;
    }

    void setNearFarPlane(float near, float far) {
        this->lightNearPlane = near;
        this->lightFarPlane = far;
    }

    float getNearPlane() const {
        return this->lightNearPlane;
    }

    float getFarPlane() const {
        return this->lightFarPlane;
    }

private:
    void _setShadowMapHandle(ShadowMapHandle handle) {
        this->_shadowHap = handle;
    }
};
}

#endif //STRATUSGFX_LIGHT_H
