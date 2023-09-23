
#ifndef STRATUSGFX_LIGHT_H
#define STRATUSGFX_LIGHT_H

#include "StratusCommon.h"
#include "StratusMath.h"
#include <algorithm>
#include <cmath>
#include <memory>
#include "StratusLog.h"
#include "StratusEngine.h"
#include "StratusTypes.h"
#include "StratusHandle.h"
#include <limits>
#include <unordered_set>

namespace stratus {
    class InfiniteLight;
    class Light;

    typedef std::shared_ptr<InfiniteLight> InfiniteLightPtr;
    typedef std::shared_ptr<Light> LightPtr;
    typedef Handle<Light> LightHandle;

    enum class LightType {
        POINTLIGHT,
        SPOTLIGHT
    };

    constexpr f32 maxLightColor = 10000.0f;
    constexpr f32 minLightColor = 0.25f;
    constexpr f32 maxAmbientIntensity = 0.02;
    constexpr f32 minAmbientIntensity = 0.001;

    // Serves as a global world light
    class InfiniteLight {
        glm::vec3 color_ = glm::vec3(1.0f);
        glm::vec3 position_ = glm::vec3(0.0f);
        Rotation rotation_;
        // Used to calculate ambient intensity based on sun orientation
        stratus::Radians rotSine_;
        f32 intensity_ = 4.0f;
        f32 ambientIntensity_ = minAmbientIntensity;
        bool enabled_ = true;
        bool runAlphaTest_ = true;
        // This is the number of rays we march per pixel to determine the final
        // atmospheric value
        int numAtmosphericSamples_ = 64;
        f32 particleDensity_ = 0.002f;
        // If > 1, then backscattered light will be greater than forwardscattered light
        f32 scatterControl_ = 0.004f; // 0.004 is roughly a G of 0.7
        glm::vec3 atmosphereColor_ = glm::vec3(1.0f);
        f32 depthBias_ = 0.0f;

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
        void SetColor(const glm::vec3 & color) { MarkChanged(); color_ = glm::max(color, glm::vec3(0.0f)); }

        const glm::vec3 & GetPosition() const { return position_; }
        void SetPosition(const glm::vec3 & position) { position_ = position; }

        const Rotation & GetRotation() const { return rotation_; }
        void SetRotation(const Rotation & rotation) { 
            MarkChanged();
            rotation_ = rotation;
            rotSine_ = stratus::sine(rotation_.x);
        }

        void OffsetRotation(const glm::vec3& offsets) {
            MarkChanged();
            Rotation rot = rotation_;
            rot.x += Degrees(offsets.x);
            rot.y += Degrees(offsets.y);
            rot.z += Degrees(offsets.z);
            SetRotation(rot);
        }

        f32 GetIntensity() const { 
            // Reduce light intensity as sun goes down
            // if (rotSine_.value() < 0.0f) {
            //     return std::max(minLightColor, intensity_ * (1.0f + rotSine_.value()));
            // }
            return intensity_; 
        }

        void SetIntensity(f32 intensity) { intensity_ = std::max(intensity, 0.0f); }

        f32 GetAmbientIntensity() const { 
            //const f32 ambient = _rotSine.value() * maxAmbientIntensity;
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
        void SetAtmosphericLightingConstants(f32 particleDensity, f32 scatterControl) {
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

        f32 GetAtmosphericParticleDensity() const {
            return particleDensity_;
        }

        f32 GetAtmosphericScatterControl() const {
            return scatterControl_;
        }

        const glm::vec3& GetAtmosphereColor() const {
            return atmosphereColor_;
        }

        f32 GetDepthBias() const {
            return depthBias_;
        }

        void SetDepthBias(const f32 bias) {
            depthBias_ = bias;
        }

        virtual InfiniteLightPtr Copy() const {
            return InfiniteLightPtr(new InfiniteLight(*this));
        }

        void MarkChanged() {
            if (INSTANCE(Engine)) {
                lastFrameChanged_ = INSTANCE(Engine)->FrameCount();
            }
        }

        bool ChangedLastFrame() const {
            u64 diff = INSTANCE(Engine)->FrameCount() - lastFrameChanged_;
            return diff == 1;
        }

        bool ChangedThisFrame() const {
            u64 diff = INSTANCE(Engine)->FrameCount() - lastFrameChanged_;
            return diff == 0;
        }

        bool ChangedWithinLastFrame() const {
            u64 diff = INSTANCE(Engine)->FrameCount() - lastFrameChanged_;
            return diff <= 1;
        }

    private:
        // Last engine frame this component was modified
        uint64_t lastFrameChanged_ = 0;
    };

    class Light {
    protected:
        glm::vec3 position_ = glm::vec3(0.0f);
        glm::vec3 color_ = glm::vec3(1.0f);
        glm::vec3 baseColor_ = color_;
        u64 lastFramePositionChanged_ = 0;
        u64 lastFrameRadiusChanged_ = 0;
        f32 intensity_ = 200.0f;
        f32 radius_ = 1.0f;
        LightHandle handle_;
        bool castsShadows_ = true;
        // If virtual we intend to use it less as a natural light and more
        // as a way of simulating bounce lighting
        bool virtualLight_ = false;
        // If true then we don't want it to be updated when dynamic entities
        // change in the scene (can still cast light, just shadows will not be updated)
        bool staticLight_ = false;

        Light(const bool virtualLight, const bool staticLight)
            : virtualLight_(virtualLight), staticLight_(staticLight) {
            handle_ = LightHandle::NextHandle();
        }

    public:
        Light(const bool staticLight) : Light(false, staticLight) {}
        virtual ~Light() = default;

        LightHandle Handle() const {
            return handle_;
        }

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
        void SetColor(f32 r, f32 g, f32 b) {
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
        void SetIntensity(f32 i) {
            if (i < 0) return;
            intensity_ = i;
            RecalcColorWithIntensity_();
            RecalcRadius_();
        }

        f32 GetIntensity() const {
            return intensity_;
        }

        // Gets radius but bounded
        virtual f32 GetRadius() const {
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
            static const f32 lightMin = 256.0 / 5;
            const glm::vec3 intensity = GetColor(); // Factors in intensity already
            const f32 Imax = std::max(intensity.x, std::max(intensity.y, intensity.z));
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
        f32 lightNearPlane = 0.1f;
        f32 lightFarPlane = 500.0f;

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

        void SetNearFarPlane(f32 nearPlane, f32 farPlane) {
            this->lightNearPlane = nearPlane;
            this->lightFarPlane = farPlane;
        }

        f32 GetNearPlane() const {
            return this->lightNearPlane;
        }

        f32 GetFarPlane() const {
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

        void SetNumShadowSamples(u32 samples) { numShadowSamples_ = samples; }
        u32 GetNumShadowSamples() const { return numShadowSamples_; }

        // This MUST be done or else the engine makes copies and it will defer
        // to PointLight instead of this and then cause horrible strange errors
        LightPtr Copy() const override {
            return LightPtr(new VirtualPointLight(*this));
        }

        virtual f32 GetRadius() const override {
            return std::max(500.0f, radius_);
        }

    private:
        u32 numShadowSamples_ = 3;
    };

    // Maintains a spatial light data structure with configurable world tile size
    struct SpatialLightMap {
        typedef std::unordered_set<LightPtr> LightContainer;

        struct SpatialLightTile {
            UnsafePtr<LightContainer> lights;

            SpatialLightTile() : lights(MakeUnsafe<LightContainer>()) {}
        };

        struct SpatialLightTileView {
            SpatialLightTileView() {}

            SpatialLightTileView(const UnsafePtr<LightContainer>& lights)
                : lights_(lights) {}

            SpatialLightTileView(const SpatialLightTile& tile)
                : SpatialLightTileView(tile.lights) {}

            const LightContainer& Lights() const {
                return *lights_;
            }

        private:
            UnsafePtr<LightContainer> lights_;
        };

        SpatialLightMap(const u32 worldTileSizeXY = 256) 
            : worldTileSizeXY_(i32(worldTileSizeXY)) {}

        // Returns the nearest neighbors in the range of [-tileOffset, tileOffset]
        template<typename Allocator>
        std::vector<SpatialLightTileView, Allocator> GetNearestTiles(const glm::vec3& origin, const Allocator& alloc, const u32 tileOffset = 1) {
            const i32 numTilesRadius = i32(tileOffset);
            std::vector<SpatialLightTileView, Allocator> result(alloc);
            result.reserve(2 * numTilesRadius);

            // Snap x/y to the nearest world tile
            const i32 startX = i32(glm::round(origin.x / f32(worldTileSizeXY_)));
            const i32 startY = i32(glm::round(origin.z / f32(worldTileSizeXY_)));

            for (i32 x = (startX - numTilesRadius); x <= (startX + numTilesRadius); ++x) {
                for (i32 y = (startY - numTilesRadius); y <= (startY + numTilesRadius); ++y) {
                    const u32 tileIndex = CalculateTileIndex(x, y);
                    auto tile = GetTile_(tileIndex);
                    if (tile->size() > 0) {
                        result.push_back(SpatialLightTileView(tile));
                    }
                }
            }

            return result;
        }

        // Returns the nearest neighbors in the range of [-tileOffset, tileOffset]
        std::vector<SpatialLightTileView> GetNearestTiles(const glm::vec3& origin, const u32 tileOffset = 1) {
            return GetNearestTiles(origin, std::allocator<SpatialLightTileView>(), tileOffset);
        }

        bool Contains(const LightHandle& handle) const {
            return lights_.find(handle) != lights_.end();
        }

        bool Contains(const LightPtr& light) const {
            return Contains(light->Handle());
        }

        i32 GetWorldTileSize() const {
            return worldTileSizeXY_;
        }

        // Inserts light if not present. If it is present, it updates its position.
        void Insert(const LightPtr& light) {
            const auto handle = light->Handle();
            auto it = lightPositions_.find(handle);
            if (it != lightPositions_.end(handle)) {
                const auto index = ConvertWorldPosToTileIndex(it->second);
                lights_.find(index)->second.lights->erase(light);
            }

            lightPositions_.insert(std::make_pair(handle, light->GetPosition()));
            const auto index = ConvertWorldPosToTileIndex(light->GetPosition());

            auto lit = lights_.find(index);
            if (lit == lights_.end()) {
                lights_.insert(std::make_pair(index, SpatialLightTile()));
                lit = lights_.find(index);
            }
            lit->second.lights->insert(light);
        }

        void Erase(const LightPtr& light) {
            auto it = lightPositions_.find(light->Handle());
            if (it != lightPositions_.end()) {
                const auto index = ConvertWorldPosToTileIndex(it->second);
                lights_.find(index)->second.lights->erase(light);
            }
            lightPositions_.erase(light->Handle());
        }

        usize Size() const {
            return lightPositions_.size();
        }

        static u32 CalculateTileIndex(const i32 tileX, const i32 tileY) {
            static constexpr i32 tilesPerRow = 32768;
            return u32(tileX + tilesPerRow) + u32(tileY + tilesPerRow) * u32(tilesPerRow);
        }

        u32 ConvertWorldPosToTileIndex(const glm::vec3& position) const {
            const i32 x = i32(glm::floor(position.x / f32(worldTileSizeXY_)));
            const i32 y = i32(glm::floor(position.z / f32(worldTileSizeXY_)));
            return CalculateTileIndex(x, y);
        }

    private:
        UnsafePtr<LightContainer> GetTile_(const u32 index) const {
            auto it = lights_.find(index);
            return it == lights_.end() ? empty_ : it->second.lights;
        }

    private:
        const UnsafePtr<LightContainer> empty_ = MakeUnsafe<LightContainer>();

        // Stores last known position
        std::unordered_map<LightHandle, glm::vec3> lightPositions_;
        // Sparse representation of the world
        std::unordered_map<u32, SpatialLightTile> lights_;
        i32 worldTileSizeXY_;
    };
}

#endif //STRATUSGFX_LIGHT_H
