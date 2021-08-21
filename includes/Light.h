
#ifndef STRATUSGFX_LIGHT_H
#define STRATUSGFX_LIGHT_H

#include "Common.h"
#include <algorithm>
#include <cmath>

namespace stratus {
enum class LightType {
    POINTLIGHT,
    SPOTLIGHT
};

const float maxLightColor = 10000.0f;

// Serves as a global world light
class InfiniteLight {
    glm::vec3 _color = glm::vec3(1.0f);
    glm::vec3 _direction = glm::vec3(0.0f, 1.0f, 0.0f);
    float _intensity = 1.0f;

public:
    const glm::vec3 & getDirection() const { return _direction; }
    void setDirection(const glm::vec3 & direction) { _direction = direction; }

    const glm::vec3 & getColor() const { return _color; }
    void setColor(glm::vec3 & color) { _color = glm::max(color, glm::vec3(0.0f)); }

    float getIntensity() const { return _intensity; }
    void setIntensity(float intensity) { _intensity = std::max(intensity, 0.0f); }
};

class Light {
    glm::vec3 _color = glm::vec3(1.0f);
    glm::vec3 _baseColor = _color;
    float _intensity = 1.0f;
    float _radius = 1.0f;
    bool _castsShadows = true;

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

    const glm::vec3& getBaseColor() const {
        return _baseColor;
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
        _baseColor = _color;
        _recalcColorWithIntensity();
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
        _recalcColorWithIntensity();
        _recalcRadius();
    }

    float getIntensity() const {
        return _intensity;
    }

    float getRadius() const {
        return _radius;
    }

    void setCastsShadows(bool enable) {
        this->_castsShadows = enable;
    }

    bool castsShadows() const {
        return this->_castsShadows;
    }

private:
    // See https://learnopengl.com/Advanced-Lighting/Deferred-Shading for the equation
    void _recalcRadius() {
        static const float lightMin = 256.0 / 5;
        const glm::vec3 intensity = getIntensity() * getColor();
        const float Imax = std::max(intensity.x, std::max(intensity.y, intensity.z));
        this->_radius = std::sqrtf(-4 * (1.0 - Imax * lightMin)) / 2;
    }

    void _recalcColorWithIntensity() {
        _color = _baseColor * _intensity;
        _color = glm::clamp(_color, glm::vec3(0.0f), glm::vec3(maxLightColor));
        _color = (_color / maxLightColor) * 30.0f;
    }
};

class PointLight : public Light {
    friend class Renderer;
    
    // ShadowMapHandle _shadowHap = -1;

    // These are used to set up the light view matrix
    float lightNearPlane = 0.1f;
    float lightFarPlane = 500.0f;

public:
    ~PointLight() override = default;

    LightType getType() const override {
        return LightType::POINTLIGHT;
    }

    // ShadowMapHandle getShadowMapHandle() const {
    //     return this->_shadowHap;
    // }

    void setNearFarPlane(float nearPlane, float farPlane) {
        this->lightNearPlane = nearPlane;
        this->lightFarPlane = farPlane;
    }

    float getNearPlane() const {
        return this->lightNearPlane;
    }

    float getFarPlane() const {
        //return this->lightFarPlane;
        return this->getRadius();
    }

private:
    // void _setShadowMapHandle(ShadowMapHandle handle) {
    //     this->_shadowHap = handle;
    // }
};
}

#endif //STRATUSGFX_LIGHT_H
