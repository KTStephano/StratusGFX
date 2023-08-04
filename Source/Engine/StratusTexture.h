#pragma once

#include <memory>
#include <vector>
#include "StratusHandle.h"
#include "StratusGpuCommon.h"
#include "StratusTypes.h"

namespace stratus {    
    class Texture;
    typedef Handle<Texture> TextureHandle;

    enum class TextureType : i32 {
        TEXTURE_2D,
        TEXTURE_2D_ARRAY,
        // Corresponds to GL_TEXTURE_CUBE_MAP
        TEXTURE_CUBE_MAP,
        TEXTURE_CUBE_MAP_ARRAY,
        // Indexed in pixel coordinates instead of texture coordinates
        TEXTURE_RECTANGLE
    };

    enum class TextureComponentFormat : i32 {
        RED,
        RGB,
        RG,
        SRGB,
        RGBA,
        SRGB_ALPHA,
        DEPTH,
        DEPTH_STENCIL
    };

    enum class TextureComponentSize : i32 {
        BITS_DEFAULT,
        BITS_8,
        BITS_16,
        BITS_32,
        BITS_11_11_10 // Only valid for float: R11G11B10_F
    };

    enum class TextureComponentType : i32 {
        INT,
        UINT,
        FLOAT
    };

    // Specifies the behavior for when coordinates under or overflow
    enum class TextureCoordinateWrapping : i32 {
        REPEAT,
        NEAREST,
        LINEAR,
        MIRRORED_REPEAT,
        CLAMP_TO_EDGE,
        CLAMP_TO_BORDER,
        MIRROR_CLAMP_TO_EDGE
    };

    enum class TextureMinificationFilter : i32 {
        NEAREST,
        LINEAR,
        NEAREST_MIPMAP_NEAREST,
        LINEAR_MIPMAP_NEAREST,
        NEAREST_MIPMAP_LINEAR,
        LINEAR_MIPMAP_LINEAR
    };

    enum class TextureMagnificationFilter : i32 {
        NEAREST,
        LINEAR
    };

    enum class TextureCompareMode : i32 {
        NONE,
        // Interpolated and clamped r texture coordinate should be compared to the value
        // in the currently bound depth texture
        COMPARE_REF_TO_TEXTURE
    };

    enum class TextureCompareFunc : i32 {
        ALWAYS,
        NEVER,
        LESS,
        LEQUAL,
        GREATER,
        GEQUAL,
        EQUAL,
        NOTEQUAL
    };

    enum class ImageTextureAccessMode : i32 {
        IMAGE_READ_ONLY,
        IMAGE_WRITE_ONLY,
        IMAGE_READ_WRITE
    };

    struct TextureConfig {
        TextureType type;
        TextureComponentFormat format;
        TextureComponentSize storage;
        TextureComponentType dataType;
        u32 width;
        u32 height;
        u32 depth;
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
        friend struct TextureMemResidencyGuard;
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

        u32 Width() const;
        u32 Height() const;
        u32 Depth() const;

        void Bind(i32 activeTexture = 0) const;
        void Unbind() const;

        void BindAsImageTexture(u32 unit, bool layered, int32_t layer, ImageTextureAccessMode access) const;

        bool Valid() const;

        // clearValue is between one and four components worth of data (or nullptr - in which case the texture is filled with 0s)
        void Clear(const i32 mipLevel, const void * clearValue) const;
        void ClearLayer(const i32 mipLevel, const i32 layer, const void * clearValue) const;

        // Gets a pointer to the underlying data (implementation-dependent)
        const void * Underlying() const;

        size_t HashCode() const;
        bool operator==(const Texture & other) const;

        // Creates a new texture and copies this texture into it
        Texture Copy(u32 newWidth, u32 newHeight) const;
        const TextureConfig & GetConfig() const;

    private:
        // Makes the texture resident in GPU memory for bindless use
        static void MakeResident_(const Texture&);
        // Removes residency
        static void MakeNonResident_(const Texture&);

    private:
        void SetHandle_(const TextureHandle);
    };

    struct TextureMemResidencyGuard {
        TextureMemResidencyGuard(const Texture&);

        TextureMemResidencyGuard(TextureMemResidencyGuard&&) noexcept;
        TextureMemResidencyGuard(const TextureMemResidencyGuard&) noexcept;

        TextureMemResidencyGuard& operator=(TextureMemResidencyGuard&&) noexcept;
        TextureMemResidencyGuard& operator=(const TextureMemResidencyGuard&) noexcept;

        ~TextureMemResidencyGuard();

    private:
        void Copy_(const TextureMemResidencyGuard&);
        void IncrementRefcount_();
        void DecrementRefcount_();

    private:
        Texture texture_ = Texture();
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