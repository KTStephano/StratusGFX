
#ifndef STRATUSGFX_LIGHT_H
#define STRATUSGFX_LIGHT_H

#include "StratusCommon.h"
#include "StratusMath.h"
#include <algorithm>
#include <cmath>
#include <memory>
#include "StratusLog.h"

namespace stratus {
    class InfiniteLight;
    class Light;

    typedef std::shared_ptr<InfiniteLight> InfiniteLightPtr;
    typedef std::shared_ptr<Light> LightPtr;

    enum class LightType {
        POINTLIGHT,
        SPOTLIGHT
    };

    constexpr float maxLightColor = 10000.0f;
    constexpr float minLightColor = 0.25f;
    constexpr float maxAmbientIntensity = 0.02;
    constexpr float minAmbientIntensity = 0.001;

    // Serves as a global world light
    class InfiniteLight {
        glm::vec3 _color = glm::vec3(1.0f);
        glm::vec3 _position = glm::vec3(0.0f);
        Rotation _rotation;
        // Used to calculate ambient intensity based on sun orientation
        stratus::Radians _rotSine;
        float _intensity = 4.0f;
        float _ambientIntensity = minAmbientIntensity;
        bool _enabled = true;

    public:
        InfiniteLight(const bool enabled = true)
            : _enabled(enabled) {}

        ~InfiniteLight() = default;

        InfiniteLight(const InfiniteLight&) = default;
        InfiniteLight(InfiniteLight&&) = default;
        InfiniteLight& operator=(const InfiniteLight&) = default;
        InfiniteLight& operator=(InfiniteLight&&) = default;

        // Get light color * intensity for use with lighting equations
        glm::vec3 getLuminance() const { return getColor() * getIntensity(); }

        const glm::vec3 & getColor() const { return _color; }
        void setColor(const glm::vec3 & color) { _color = glm::max(color, glm::vec3(0.0f)); }

        const glm::vec3 & getPosition() const { return _position; }
        void setPosition(const glm::vec3 & position) { _position = position; }

        const Rotation & getRotation() const { return _rotation; }
        void setRotation(const Rotation & rotation) { 
            _rotation = rotation;
            _rotSine = stratus::sine(_rotation.x);
        }

        void offsetRotation(const glm::vec3& offsets) {
            Rotation rot = _rotation;
            rot.x += Degrees(offsets.x);
            rot.y += Degrees(offsets.y);
            rot.z += Degrees(offsets.z);
            setRotation(rot);
        }

        float getIntensity() const { 
            // Reduce light intensity as sun goes down
            if (_rotSine.value() < 0.0f) {
                return std::max(minLightColor, _intensity * (1.0f + _rotSine.value()));
            }
            return _intensity; 
        }

        void setIntensity(float intensity) { _intensity = std::max(intensity, 0.0f); }

        float getAmbientIntensity() const { 
            //const float ambient = _rotSine.value() * maxAmbientIntensity;
            //return std::min(maxAmbientIntensity, std::max(ambient, minAmbientIntensity));
            return minAmbientIntensity;
        }

        bool getEnabled() const { return _enabled; }
        void setEnabled(const bool e) { _enabled = e; }

        virtual InfiniteLightPtr Copy() const {
            return InfiniteLightPtr(new InfiniteLight(*this));
        }
    };

    class Light {
        glm::vec3 _color = glm::vec3(1.0f);
        glm::vec3 _baseColor = _color;
        float _intensity = 1.0f;
        float _radius = 1.0f;
        bool _castsShadows = true;
        // If virtual we intend to use it less as a natural light and more
        // as a way of simulating bounce lighting
        bool _virtualLight = false;

    public:
        glm::vec3 position = glm::vec3(0.0f);

        Light(const bool virtualLight = false)
            : _virtualLight(virtualLight) {}
        
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

        void setColor(const glm::vec3& color) {
            setColor(color.r, color.g, color.b);
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

        // If true then the light will be invisible when the sun is not overhead - 
        // useful for brightening up directly-lit scenes without Static or RT GI
        bool IsVirtualLight() const { return _virtualLight; }

        virtual LightPtr Copy() const = 0;

    private:
        // See https://learnopengl.com/Advanced-Lighting/Deferred-Shading for the equation
        void _recalcRadius() {
            static const float lightMin = 256.0 / 5;
            const glm::vec3 intensity = getIntensity() * getColor();
            const float Imax = std::max(intensity.x, std::max(intensity.y, intensity.z));
            //this->_radius = sqrtf(-4 * (1.0 - Imax * lightMin)) / 2;
            this->_radius = sqrtf(Imax * lightMin - 1.0f) * 2.0f;
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

    protected:
        PointLight(const bool virtualLight) 
            : Light(virtualLight) {}

    public:
        PointLight() : PointLight(false) {}

        virtual ~PointLight() = default;

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

        LightPtr Copy() const override {
            return LightPtr(new PointLight(*this));
        }

    private:
        // void _setShadowMapHandle(ShadowMapHandle handle) {
        //     this->_shadowHap = handle;
        // }
    };

    class VirtualPointLight : public PointLight {
        friend class Renderer;

    public:
        VirtualPointLight() : PointLight(/* virtualLight = */ true) {}
        virtual ~VirtualPointLight() = default;
    };
}

#endif //STRATUSGFX_LIGHT_H
