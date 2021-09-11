#pragma once

#include <memory>
#include "StratusHandle.h"

namespace stratus {    
    class Texture;
    typedef Handle<Texture> TextureHandle;

    enum class TextureType : int {
        TEXTURE_2D,
        TEXTURE_2D_ARRAY,
        TEXTURE_3D
    };

    enum class TextureComponentFormat : int {
        RED,
        RGB,
        SRGB,
        RGBA,
        SRGB_ALPHA,
        DEPTH,
        DEPTH_STENCIL
    };

    enum class TextureComponentSize : int {
        BITS_DEFAULT,
        BITS_8,
        BITS_16,
        BITS_32
    };

    enum class TextureComponentType : int {
        INT,
        UINT,
        FLOAT
    };

    // Specifies the behavior for when coordinates under or overflow
    enum class TextureCoordinateWrapping : int {
        REPEAT,
        NEAREST,
        LINEAR,
        MIRRORED_REPEAT,
        CLAMP_TO_EDGE,
        MIRROR_CLAMP_TO_EDGE
    };

    enum class TextureMinificationFilter : int {
        NEAREST,
        LINEAR,
        NEAREST_MIPMAP_NEAREST,
        LINEAR_MIPMAP_NEAREST,
        NEAREST_MIPMAP_LINEAR,
        LINEAR_MIPMAP_LINEAR
    };

    enum class TextureMagnificationFilter : int {
        NEAREST,
        LINEAR
    };

    enum class TextureCompareMode : int {
        NONE,
        // Interpolated and clamped r texture coordinate should be compared to the value
        // in the currently bound depth texture
        COMPARE_REF_TO_TEXTURE
    };

    enum class TextureCompareFunc : int {
        ALWAYS,
        NEVER,
        LESS,
        LEQUAL,
        GREATER,
        GEQUAL,
        EQUAL,
        NOTEQUAL
    };

    struct TextureConfig {
        TextureType type;
        TextureComponentFormat format;
        TextureComponentSize storage;
        TextureComponentType dataType;
        uint32_t width;
        uint32_t height;
        uint32_t depth;
        bool generateMipMaps;
    };

    class TextureImpl;
    class Texture {
        friend class ResourceManager;
        // Underlying implementation which may change from platform to platform
        std::shared_ptr<TextureImpl> _impl;

        Texture(std::shared_ptr<TextureImpl>) {}

    public:
        Texture();
        Texture(const TextureConfig & config, const void * data, bool initHandle = true);
        ~Texture();

        Texture(const Texture &) = default;
        Texture(Texture &&) = default;
        Texture & operator=(const Texture &) = default;
        Texture & operator=(Texture &&) = default;

        void setCoordinateWrapping(TextureCoordinateWrapping);
        void setMinMagFilter(TextureMinificationFilter, TextureMagnificationFilter);
        void setTextureCompare(TextureCompareMode, TextureCompareFunc);

        TextureType type() const;
        TextureComponentFormat format() const;
        TextureHandle handle() const;

        uint32_t width() const;
        uint32_t height() const;
        uint32_t depth() const;

        void bind(int activeTexture = 0) const;
        void unbind() const;

        bool valid() const;

        // Gets a pointer to the underlying data (implementation-dependent)
        const void * underlying() const;

        size_t hashCode() const;
        bool operator==(const Texture & other) const;

        // Creates a new texture and copies this texture into it
        Texture copy(uint32_t newWidth, uint32_t newHeight);
        const TextureConfig & getConfig() const;

    private:
        void _setHandle(const TextureHandle);
    };
}

namespace std {
    template<>
    struct hash<stratus::Texture> {
        size_t operator()(const stratus::Texture & tex) const {
            return tex.hashCode();
        }
    };
}