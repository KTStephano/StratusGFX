
#ifndef STRATUSGFX_LIGHT_H
#define STRATUSGFX_LIGHT_H

#include "StratusCommon.h"
#include "StratusMath.h"
#include <algorithm>
#include <cmath>
#include <memory>
#include "StratusLog.h"
#include "StratusEngine.h"

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
        glm::vec3 color_ = glm::vec3(1.0f);
        glm::vec3 position_ = glm::vec3(0.0f);
        Rotation rotation_;
        // Used to calculate ambient intensity based on sun orientation
        stratus::Radians rotSine_;
        float intensity_ = 4.0f;
        float ambientIntensity_ = minAmbientIntensity;
        bool enabled_ = true;
        bool runAlphaTest_ = true;
        // This is the number of rays we march per pixel to determine the final
        // atmospheric value
        int numAtmosphericSamples_ = 64;
        float particleDensity_ = 0.002f;
        // If > 1, then backscattered light will be greater than forwardscattered light
        float scatterControl_ = 0.004f; // 0.004 is roughly a G of 0.7
        glm::vec3 atmosphereColor_ = glm::vec3(1.0f);
        float depthBias_ = 0.0f;

    public:
        InfiniteLight(const bool enabled = true)
            : enabled_(enabled) {}

        ~InfiniteLight() = default;

        InfiniteLight(const InfiniteLight&) = default;
        InfiniteLight(InfiniteLight&&) = default;
        InfiniteLight& operator=(const InfiniteLight&) = default;
        InfiniteLight& operator=(InfiniteLight&&) = default;

        // Get light color * intensity for use with lighting equations
        glm::vec3 GetLuminance() const { return GetColor() * GetIntensity(); }

        const glm::vec3 & GetColor() const { return color_; }
        void SetColor(const glm::vec3 & color) { color_ = glm::max(color, glm::vec3(0.0f)); }

        const glm::vec3 & GetPosition() const { return position_; }
        void SetPosition(const glm::vec3 & position) { position_ = position; }

        const Rotation & GetRotation() const { return rotation_; }
        void SetRotation(const Rotation & rotation) { 
            rotation_ = rotation;
            rotSine_ = stratus::sine(rotation_.x);
        }

        void OffsetRotation(const glm::vec3& offsets) {
            Rotation rot = rotation_;
            rot.x += Degrees(offsets.x);
            rot.y += Degrees(offsets.y);
            rot.z += Degrees(offsets.z);
            SetRotation(rot);
        }

        float GetIntensity() const { 
            // Reduce light intensity as sun goes down
            // if (rotSine_.value() < 0.0f) {
            //     return std::max(minLightColor, intensity_ * (1.0f + rotSine_.value()));
            // }
            return intensity_; 
        }

        void SetIntensity(float intensity) { intensity_ = std::max(intensity, 0.0f); }

        float GetAmbientIntensity() const { 
            //const float ambient = _rotSine.value() * maxAmbientIntensity;
            //return std::min(maxAmbientIntensity, std::max(ambient, minAmbientIntensity));
            return minAmbientIntensity;
        }

        bool GetEnabled() const { return enabled_; }
        void SetEnabled(const bool e) { enabled_ = e; }

        // Enables alpha testing during cascaded shadow map creation - some scenes don't work
        // as well with this enabled
        void SetAlphaTest(const bool enabled) { runAlphaTest_ = enabled; }
        bool GetAlphaTest() const { return runAlphaTest_; }

        // If scatterControl > 1, then backscattered light will be greater than forwardscattered light
        void SetAtmosphericLightingConstants(float particleDensity, float scatterControl) {
            particleDensity_ = std::max(0.0f, std::min(particleDensity, 1.0f));
            scatterControl_ = std::max(0.0f, scatterControl);
        }

        void SetAtmosphereColor(const glm::vec3& color) {
            atmosphereColor_ = color;
        }

        // Number of rays that we march per pixel to determine final atmospheric value
        void SetNumAtmosphericSamplesPerPixel(const int numSamples) {
            numAtmosphericSamples_ = numSamples;
        }

        int GetAtmosphericNumSamplesPerPixel() const {
            return numAtmosphericSamples_;
        }

        float GetAtmosphericParticleDensity() const {
            return particleDensity_;
        }

        float GetAtmosphericScatterControl() const {
            return scatterControl_;
        }

        const glm::vec3& GetAtmosphereColor() const {
            return atmosphereColor_;
        }

        float GetDepthBias() const {
            return depthBias_;
        }

        void SetDepthBias(const float bias) {
            depthBias_ = bias;
        }

        virtual InfiniteLightPtr Copy() const {
            return InfiniteLightPtr(new InfiniteLight(*this));
        }
    };

    class Light {
    protected:
        glm::vec3 position_ = glm::vec3(0.0f);
        glm::vec3 color_ = glm::vec3(1.0f);
        glm::vec3 baseColor_ = color_;
        uint64_t lastFramePositionChanged_ = 0;
        uint64_t lastFrameRadiusChanged_ = 0;
        float intensity_ = 200.0f;
        float radius_ = 1.0f;
        bool castsShadows_ = true;
        // If virtual we intend to use it less as a natural light and more
        // as a way of simulating bounce lighting
        bool virtualLight_ = false;
        // If true then we don't want it to be updated when dynamic entities
        // change in the scene (can still cast light, just shadows will not be updated)
        bool staticLight_ = false;

        Light(const bool virtualLight, const bool staticLight)
            : virtualLight_(virtualLight), staticLight_(staticLight) {}

    public:
        Light(const bool staticLight) : Light(false, staticLight) {}
        virtual ~Light() = default;

        const glm::vec3& GetPosition() const {
            return position_;
        }

        void SetPosition(const glm::vec3& position) {
            position_ = position;
            lastFramePositionChanged_ = INSTANCE(Engine)->FrameCount();
        }

        bool PositionChangedWithinLastFrame() const {
            auto diff = INSTANCE(Engine)->FrameCount() - lastFramePositionChanged_;
            return diff <= 1;
        }

        /**
         * @return type of point light so that the renderer knows
         *      how to deal with it
         */
        virtual LightType GetType() const = 0;

        const glm::vec3 & GetColor() const {
            return color_;
        }

        const glm::vec3& GetBaseColor() const {
            return baseColor_;
        }

        /**
         * Sets the color of the light where the scale
         * is not from [0.0, 1.0] but instead can be any
         * number > 0.0 for each color component. To make this
         * work, HDR support is required.
         */
        void SetColor(float r, float g, float b) {
            r = std::max(0.0f, r);
            g = std::max(0.0f, g);
            b = std::max(0.0f, b);
            color_ = glm::vec3(r, g, b);
            baseColor_ = color_;
            RecalcColorWithIntensity_();
            //_recalcRadius();
        }

        void SetColor(const glm::vec3& color) {
            SetColor(color.r, color.g, color.b);
        }

        /**
         * A light's color values can all be on the range of
         * [0.0, 1.0], but the intensity specifies how strong it
         * should be.
         * @param i
         */
        void SetIntensity(float i) {
            if (i < 0) return;
            intensity_ = i;
            RecalcColorWithIntensity_();
            RecalcRadius_();
        }

        float GetIntensity() const {
            return intensity_;
        }

        // Gets radius but bounded
        virtual float GetRadius() const {
            return std::max(150.0f, radius_);
        }

        bool RadiusChangedWithinLastFrame() const {
            auto diff = INSTANCE(Engine)->FrameCount() - lastFrameRadiusChanged_;
            return diff <= 1;

        }

        void SetCastsShadows(bool enable) {
            this->castsShadows_ = enable;
        }

        bool CastsShadows() const {
            return this->castsShadows_;
        }

        // If true then the light will be invisible when the sun is not overhead - 
        // useful for brightening up directly-lit scenes without Static or RT GI
        bool IsVirtualLight() const { return virtualLight_; }
        bool IsStaticLight()  const { return staticLight_; }

        virtual LightPtr Copy() const = 0;

    private:
        // See https://learnopengl.com/Advanced-Lighting/Deferred-Shading for the equation
        void RecalcRadius_() {
            static const float lightMin = 256.0 / 5;
            const glm::vec3 intensity = GetColor(); // Factors in intensity already
            const float Imax = std::max(intensity.x, std::max(intensity.y, intensity.z));
            //_radius = sqrtf(4.0f * (Imax * lightMin - 1.0f)) / 2.0f;
            radius_ = sqrtf(Imax * lightMin - 1.0f) * 2.0f;
            lastFrameRadiusChanged_ = INSTANCE(Engine)->FrameCount();
        }

        void RecalcColorWithIntensity_() {
            color_ = baseColor_ * intensity_;
            color_ = glm::clamp(color_, glm::vec3(0.0f), glm::vec3(maxLightColor));
            // _color = (_color / maxLightColor) * 30.0f;
        }
    };

    class PointLight : public Light {
        friend class Renderer;
        
        // ShadowMapHandle _shadowHap = -1;

        // These are used to set up the light view matrix
        float lightNearPlane = 0.1f;
        float lightFarPlane = 500.0f;

    protected:
        PointLight(const bool virtualLight, const bool staticLight) 
            : Light(virtualLight, staticLight) {}

    public:
        PointLight(const bool staticLight) : PointLight(false, staticLight) {}

        virtual ~PointLight() = default;

        LightType GetType() const override {
            return LightType::POINTLIGHT;
        }

        // ShadowMapHandle getShadowMapHandle() const {
        //     return this->_shadowHap;
        // }

        void SetNearFarPlane(float nearPlane, float farPlane) {
            this->lightNearPlane = nearPlane;
            this->lightFarPlane = farPlane;
        }

        float GetNearPlane() const {
            return this->lightNearPlane;
        }

        float GetFarPlane() const {
            //return this->lightFarPlane;
            return this->GetRadius();
        }

        LightPtr Copy() const override {
            return LightPtr(new PointLight(*this));
        }

    private:
        // void _setShadowMapHandle(ShadowMapHandle handle) {
        //     this->_shadowHap = handle;
        // }
    };

    // If you create a VPL and do not set a color for it, it will automatically
    // inherit the color of the sun at each frame. Once a manual color is set this automatic
    // changing will be disabled.
    class VirtualPointLight : public PointLight {
        friend class Renderer;

    public:
        VirtualPointLight() : PointLight(/* virtualLight = */ true, /* staticLight = */ true) {}
        virtual ~VirtualPointLight() = default;

        void SetNumShadowSamples(uint32_t samples) { numShadowSamples_ = samples; }
        uint32_t GetNumShadowSamples() const { return numShadowSamples_; }

        // This MUST be done or else the engine makes copies and it will defer
        // to PointLight instead of this and then cause horrible strange errors
        LightPtr Copy() const override {
            return LightPtr(new VirtualPointLight(*this));
        }

        virtual float GetRadius() const override {
            return std::max(500.0f, radius_);
        }

    private:
        uint32_t numShadowSamples_ = 3;
    };
}

#endif //STRATUSGFX_LIGHT_H
