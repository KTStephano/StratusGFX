#pragma once

#include <memory>
#include <vector>
#include "StratusHandle.h"
#include "StratusGpuCommon.h"

namespace stratus {    
    class Texture;
    typedef Handle<Texture> TextureHandle;

    enum class TextureType : int {
        TEXTURE_2D,
        TEXTURE_2D_ARRAY,
        // Corresponds to GL_TEXTURE_CUBE_MAP
        TEXTURE_CUBE_MAP,
        TEXTURE_CUBE_MAP_ARRAY,
        // Indexed in pixel coordinates instead of texture coordinates
        TEXTURE_RECTANGLE
    };

    enum class TextureComponentFormat : int {
        RED,
        RGB,
        RG,
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
        BITS_32,
        BITS_11_11_10 // Only valid for float: R11G11B10_F
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
        CLAMP_TO_BORDER,
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

    enum class ImageTextureAccessMode : int {
        IMAGE_READ_ONLY,
        IMAGE_WRITE_ONLY,
        IMAGE_READ_WRITE
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

    struct TextureData {
        const void * data;
        TextureData(const void * data = nullptr) : data(data) {}
    };

    typedef std::vector<TextureData> TextureArrayData;
    const TextureArrayData NoTextureData = TextureArrayData{};

    class TextureImpl;
    class Texture {
        friend class ResourceManager;
        friend class TextureImpl;
        // Underlying implementation which may change from platform to platform
        std::shared_ptr<TextureImpl> impl_;

        Texture(std::shared_ptr<TextureImpl>) {}

    public:
        Texture();
        Texture(const TextureConfig & config, const TextureArrayData& data, bool initHandle = true);
        ~Texture();

        Texture(const Texture &) = default;
        Texture(Texture &&) = default;
        Texture & operator=(const Texture &) = default;
        Texture & operator=(Texture &&) = default;

        void SetCoordinateWrapping(TextureCoordinateWrapping);
        void SetMinMagFilter(TextureMinificationFilter, TextureMagnificationFilter);
        void SetTextureCompare(TextureCompareMode, TextureCompareFunc);

        TextureType Type() const;
        TextureComponentFormat Format() const;
        TextureHandle Handle() const;
        
        // 64 bit handle representing the texture within the graphics driver
        GpuTextureHandle GpuHandle() const;
        // Makes the texture resident in GPU memory for bindless use
        static void MakeResident(const Texture&);
        // Removes residency
        static void MakeNonResident(const Texture&);

        uint32_t Width() const;
        uint32_t Height() const;
        uint32_t Depth() const;

        void Bind(int activeTexture = 0) const;
        void Unbind() const;

        void BindAsImageTexture(uint32_t unit, bool layered, int32_t layer, ImageTextureAccessMode access) const;

        bool Valid() const;

        // clearValue is between one and four components worth of data (or nullptr - in which case the texture is filled with 0s)
        void Clear(const int mipLevel, const void * clearValue) const;
        void ClearLayer(const int mipLevel, const int layer, const void * clearValue) const;

        // Gets a pointer to the underlying data (implementation-dependent)
        const void * Underlying() const;

        size_t HashCode() const;
        bool operator==(const Texture & other) const;

        // Creates a new texture and copies this texture into it
        Texture Copy(uint32_t newWidth, uint32_t newHeight) const;
        const TextureConfig & GetConfig() const;

    private:
        void SetHandle_(const TextureHandle);
    };
}

namespace std {
    template<>
    struct hash<stratus::Texture> {
        size_t operator()(const stratus::Texture & tex) const {
            return tex.HashCode();
        }
    };
}