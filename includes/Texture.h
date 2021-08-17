#pragma once

#include <memory>

namespace stratus {    
    enum class TextureType : int {
        TEXTURE_2D,
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

    struct TextureConfig {
        TextureType type;
        TextureComponentFormat format;
        TextureComponentSize storage;
        TextureComponentType dataType;
        uint32_t width;
        uint32_t height;
        bool generateMipMaps;
    };

    class TextureImpl;
    class Texture {
        // Underlying implementation which may change from platform to platform
        std::shared_ptr<TextureImpl> _impl;

    public:
        Texture();
        Texture(const TextureConfig & config, const void * data);
        ~Texture();

        Texture(const Texture &) = default;
        Texture(Texture &&) = default;
        Texture & operator=(const Texture &) = default;
        Texture & operator=(Texture &&) = default;

        void setCoordinateWrapping(TextureCoordinateWrapping);
        void setMinMagFilter(TextureMinificationFilter, TextureMagnificationFilter);

        TextureType type() const;
        TextureComponentFormat format() const;

        uint32_t width() const;
        uint32_t height() const;
    };
}