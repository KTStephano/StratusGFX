#include "StratusTexture.h"
#include "StratusLog.h"
#include <GL/gl3w.h>
#include <exception>
#include <unordered_set>
#include <iostream>
#include "StratusApplicationThread.h"
#include "StratusGraphicsDriver.h"

namespace stratus {
    static const void* CastTexDataToPtr(const TextureArrayData& data, const size_t offset) {
        if (!data.size()) return nullptr;
        return data[offset].data;
    }

    TextureType type;
    TextureComponentFormat format;
    TextureComponentSize storage;
    TextureComponentType dataType;
    u32 width;
    u32 height;
    u32 depth;
    bool generateMipMaps;

    static std::string ConvertTextureConfigToString(const TextureConfig& config) {
        std::string result = "TextureConfig{";

        switch (config.type) {
        case TextureType::TEXTURE_2D: result += "TEXTURE_2D, "; break;
        case TextureType::TEXTURE_2D_ARRAY: result += "TEXTURE_2D_ARRAY, "; break;
        case TextureType::TEXTURE_CUBE_MAP: result += "TEXTURE_CUBE_MAP, "; break;
        case TextureType::TEXTURE_RECTANGLE: result += "TEXTURE_RECTANGLE, "; break;
        case TextureType::TEXTURE_CUBE_MAP_ARRAY: result += "TEXTURE_CUBE_MAP_ARRAY, "; break;
        case TextureType::TEXTURE_3D: result += "TEXTURE_3D, "; break;
        default: throw std::exception();
        }

        switch (config.format) {
        case TextureComponentFormat::RED: result += "RED, "; break;
        case TextureComponentFormat::RGB: result += "RGB, "; break;
        case TextureComponentFormat::RG: result += "RG, "; break;
        case TextureComponentFormat::SRGB: result += "SRGB, "; break;
        case TextureComponentFormat::RGBA: result += "RGBA, "; break;
        case TextureComponentFormat::SRGB_ALPHA: result += "SRGB_ALPHA, "; break;
        case TextureComponentFormat::DEPTH: result += "DEPTH, "; break;
        case TextureComponentFormat::DEPTH_STENCIL: result += "DEPTH_STENCIL, "; break;
        default: throw std::exception();
        }

        switch (config.storage) {
        case TextureComponentSize::BITS_DEFAULT: result += "BITS_DEFAULT, "; break;
        case TextureComponentSize::BITS_8: result += "BITS_8, "; break;
        case TextureComponentSize::BITS_16: result += "BITS_16, "; break;
        case TextureComponentSize::BITS_32: result += "BITS_32, "; break;
        case TextureComponentSize::BITS_11_11_10: result += "BITS_11_11_10, "; break;
        default: throw std::exception();
        }

        switch (config.dataType) {
        case TextureComponentType::INT_NORM: result += "INT_NORM, "; break;
        case TextureComponentType::UINT_NORM: result += "UINT_NORM, "; break;
        case TextureComponentType::INT: result += "INT, "; break;
        case TextureComponentType::UINT: result += "UINT, "; break;
        case TextureComponentType::FLOAT: result += "FLOAT, "; break;
        default: throw std::exception();
        }

        result = result + std::to_string(config.width) + ", " +
            std::to_string(config.height) + ", " +
            std::to_string(config.depth) + ", " +
            std::to_string(config.generateMipMaps) + ", " +
            std::to_string(config.virtualTexture) + "}";

        return result;
    }

    class TextureImpl {
        friend struct TextureMemResidencyGuard;

        GLuint texture_;
        TextureConfig config_;
        TextureHandle handle_;
        i32 memRefcount_ = 0;

    public:
        TextureImpl(const TextureConfig& config, const TextureArrayData& data, bool initHandle) {
            if (initHandle) {
                handle_ = TextureHandle::NextHandle();
            }

            //glGenTextures(1, &texture_);
            glCreateTextures(_convertTexture(config.type), 1, &texture_);

            config_ = config;

            if (config_.virtualTexture) {
                glTextureParameteri(texture_, GL_TEXTURE_SPARSE_ARB, GL_TRUE);
                if (config.type == TextureType::TEXTURE_2D || config.type == TextureType::TEXTURE_RECTANGLE) {
                    glTextureStorage2D(
                        texture_,
                        1,
                        _convertInternalFormatPrecise(config.format, config.storage, config.dataType),
                        config.width,
                        config.height
                    );
                }
                else if (config.type == TextureType::TEXTURE_2D_ARRAY) {
                    glTextureStorage3D(
                        texture_,
                        1,
                        _convertInternalFormatPrecise(config.format, config.storage, config.dataType),
                        config.width,
                        config.height,
                        config.depth
                    );
                }
                else {
                    throw std::runtime_error("Unsupported virtual (sparse) texture type");
                }
            }
            else {
                bind(0);
                // Use tightly packed data
                // See https://stackoverflow.com/questions/19023397/use-glteximage2d-draw-6363-image
                // See https://registry.khronos.org/OpenGL-Refpages/es1.1/xhtml/glPixelStorei.xml
                glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
                if (config.type == TextureType::TEXTURE_2D || config.type == TextureType::TEXTURE_RECTANGLE) {
                    glTexImage2D(_convertTexture(config.type), // target
                        0, // level 
                        _convertInternalFormat(config.format, config.storage, config.dataType), // internal format (e.g. RGBA16F)
                        config.width,
                        config.height,
                        0,
                        _convertFormat(config.format, config.dataType), // format (e.g. RGBA)
                        _convertType(config.dataType, config.storage), // type (e.g. FLOAT)
                        CastTexDataToPtr(data, 0)
                    );

                    // Set anisotropic filtering
                    auto maxAnisotropy = GraphicsDriver::GetConfig().maxAnisotropy;
                    //maxAnisotropy = maxAnisotropy > 2.0f ? 2.0f : maxAnisotropy;
                    glTexParameterf(_convertTexture(config.type), GL_TEXTURE_MAX_ANISOTROPY, maxAnisotropy);
                }
                else if (config.type == TextureType::TEXTURE_2D_ARRAY ||
                    config.type == TextureType::TEXTURE_CUBE_MAP_ARRAY ||
                    config.type == TextureType::TEXTURE_3D) {

                    if ((config.type != TextureType::TEXTURE_3D && config.width != config.height) || config.depth < 1) {
                        throw std::runtime_error("Unable to create array texture");
                    }

                    // Cube map array depth is in terms of faces, so it should be desired depth * 6
                    // if (config.type == TextureType::TEXTURE_CUBE_MAP_ARRAY && (config.depth % 6) != 0) {
                    //     throw std::runtime_error("Depth must be divisible by 6 for cube map arrays");
                    // }

                    u32 depth = config.depth;
                    // Cube map array depth is in terms of faces, so it should be desired depth * 6
                    if (config.type == TextureType::TEXTURE_CUBE_MAP_ARRAY) {
                        depth *= 6;
                    }

                    //STRATUS_LOG << (_convertTexture(config.type) == GL_TEXTURE_CUBE_MAP_ARRAY) << ", " << config.width << ", " << config.height << ", " << config.depth << std::endl;

                    // See: https://johanmedestrom.wordpress.com/2016/03/18/opengl-cascaded-shadow-maps/
                    // for an example of glTexImage3D
                    glTexImage3D(
                        _convertTexture(config.type), // target
                        0, // level 
                        _convertInternalFormat(config.format, config.storage, config.dataType), // internal format (e.g. RGBA16F)
                        config.width,
                        config.height,
                        depth,
                        0,
                        _convertFormat(config.format, config.dataType), // format (e.g. RGBA)
                        _convertType(config.dataType, config.storage), // type (e.g. FLOAT)
                        CastTexDataToPtr(data, 0)
                    );
                    //STRATUS_LOG << (_convertInternalFormat(config.format, config.storage, config.dataType) == GL_R16I) << std::endl;
                }
                else if (config.type == TextureType::TEXTURE_CUBE_MAP) {
                    if (config.width != config.height) {
                        throw std::runtime_error("Unable to create cube map texture");
                    }

                    for (i32 face = 0; face < 6; ++face) {
                        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + face,
                            0,
                            _convertInternalFormat(config.format, config.storage, config.dataType),
                            config.width,
                            config.height,
                            0,
                            _convertFormat(config.format, config.dataType),
                            _convertType(config.dataType, config.storage),
                            CastTexDataToPtr(data, (const size_t)face)
                        );
                    }
                }
                else {
                    throw std::runtime_error("Unknown texture type specified");
                }

                unbind(0);
            }

            // Mipmaps aren't generated for rectangle textures
            if (config.generateMipMaps && config.type != TextureType::TEXTURE_RECTANGLE) glGenerateTextureMipmap(texture_);

            STRATUS_LOG << ConvertTextureConfigToString(config) << std::endl;
        }

        ~TextureImpl() {
            if (ApplicationThread::Instance()->CurrentIsApplicationThread()) {
                glDeleteTextures(1, &texture_);
            }
            else {
                auto texture = texture_;
                ApplicationThread::Instance()->Queue([texture]() { GLuint tex = texture; glDeleteTextures(1, &tex); });
            }
        }

        // No copying
        TextureImpl(const TextureImpl&) = delete;
        TextureImpl(TextureImpl&&) = delete;
        TextureImpl& operator=(const TextureImpl&) = delete;
        TextureImpl& operator=(TextureImpl&&) = delete;

        void setCoordinateWrapping(TextureCoordinateWrapping wrap) {
            if (config_.type == TextureType::TEXTURE_RECTANGLE && (wrap != TextureCoordinateWrapping::CLAMP_TO_BORDER && wrap != TextureCoordinateWrapping::CLAMP_TO_EDGE)) {
                STRATUS_ERROR << "Texture_Rectangle ONLY supports clamp to edge and clamp to border" << std::endl;
                throw std::runtime_error("Invalid Texture_Rectangle coordinate wrapping");
            }

            bind(0);
            glTexParameteri(_convertTexture(config_.type), GL_TEXTURE_WRAP_S, _convertTextureCoordinateWrapping(wrap));
            glTexParameteri(_convertTexture(config_.type), GL_TEXTURE_WRAP_T, _convertTextureCoordinateWrapping(wrap));
            // Support third dimension for cube maps
            if (config_.type == TextureType::TEXTURE_3D ||
                config_.type == TextureType::TEXTURE_CUBE_MAP ||
                config_.type == TextureType::TEXTURE_CUBE_MAP_ARRAY) {

                glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, _convertTextureCoordinateWrapping(wrap));
            }

            unbind(0);
        }

        void setMinMagFilter(TextureMinificationFilter min, TextureMagnificationFilter mag) {
            bind(0);
            glTexParameteri(_convertTexture(config_.type), GL_TEXTURE_MIN_FILTER, _convertTextureMinFilter(min));
            glTexParameteri(_convertTexture(config_.type), GL_TEXTURE_MAG_FILTER, _convertTextureMagFilter(mag));
            unbind(0);
        }

        void setTextureCompare(TextureCompareMode mode, TextureCompareFunc func) {
            bind(0);
            glTexParameteri(_convertTexture(config_.type), GL_TEXTURE_COMPARE_MODE, _convertTextureCompareMode(mode));
            glTexParameterf(_convertTexture(config_.type), GL_TEXTURE_COMPARE_FUNC, _convertTextureCompareFunc(func));
            unbind(0);
        }

        void setHandle(const TextureHandle handle) {
            handle_ = handle;
        }

        void Clear(const i32 mipLevel, const void* clearValue) const {
            glClearTexImage(
                texture_,
                mipLevel,
                _convertFormat(config_.format, config_.dataType), // format (e.g. RGBA)
                _convertType(config_.dataType, config_.storage), // type (e.g. FLOAT))
                clearValue
            );
        }

        // See https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glClearTexSubImage.xhtml
        // for information about how to handle different texture types.
        //
        // This does not work for compressed textures or texture buffers.
        void clearLayer(const i32 mipLevel, const i32 layer, const void* clearValue) const {
            if (type() == TextureType::TEXTURE_2D || type() == TextureType::TEXTURE_RECTANGLE) {
                Clear(mipLevel, clearValue);
            }
            else {
                // For cube maps layers are interpreted as layer-faces, meaning divisible by 6
                const i32 multiplier = type() == TextureType::TEXTURE_CUBE_MAP_ARRAY ? 6 : 1;
                const i32 xoffset = 0, yoffset = 0;
                const i32 zoffset = layer * multiplier;
                const i32 depth = multiplier; // number of layers to clear which for a cubemap is 6
                glClearTexSubImage(
                    texture_,
                    mipLevel,
                    xoffset,
                    yoffset,
                    zoffset,
                    width(),
                    height(),
                    depth,
                    _convertFormat(config_.format, config_.dataType), // format (e.g. RGBA)
                    _convertType(config_.dataType, config_.storage), // type (e.g. FLOAT))
                    clearValue
                );
            }
        }

        void ClearLayerRegion(
            const i32 mipLevel,
            const i32 layer,
            const i32 xoffset,
            const i32 yoffset,
            const i32 width,
            const i32 height,
            const void* clearValue) const {

            // For cube maps layers are interpreted as layer-faces, meaning divisible by 6
            const i32 multiplier = type() == TextureType::TEXTURE_CUBE_MAP_ARRAY ? 6 : 1;
            const i32 zoffset = layer * multiplier;
            const i32 depth = multiplier; // number of layers to clear which for a cubemap is 6
            glClearTexSubImage(
                texture_,
                mipLevel,
                xoffset,
                yoffset,
                zoffset,
                width,
                height,
                depth,
                _convertFormat(config_.format, config_.dataType), // format (e.g. RGBA)
                _convertType(config_.dataType, config_.storage), // type (e.g. FLOAT))
                clearValue
            );
        }

        TextureType type() const { return config_.type; }
        TextureComponentFormat format() const { return config_.format; }
        TextureHandle handle() const { return handle_; }

        // These cause RenderDoc to disable frame capture... super unfortunate
        GpuTextureHandle GpuHandle() const {
            auto gpuHandle = glGetTextureHandleARB(texture_);
            if (gpuHandle == 0) {
                throw std::runtime_error("GPU texture handle is null");
            }
            //STRATUS_LOG << "GPU HANDLE: " << gpuHandle << std::endl;
            return (GpuTextureHandle)gpuHandle;
        }

        static void MakeResident(const Texture& texture) {
            if (texture.impl_ == nullptr) return;
            glMakeTextureHandleResidentARB((GLuint64)texture.GpuHandle());
        }

        static void MakeNonResident(const Texture& texture) {
            if (texture.impl_ == nullptr) return;
            glMakeTextureHandleNonResidentARB((GLuint64)texture.GpuHandle());
        }

        u32 width() const { return config_.width; }
        u32 height() const { return config_.height; }
        u32 depth() const { return config_.depth; }
        void* Underlying() const { return (void*)&texture_; }

        void CommitOrUncommitVirtualPage(u32 xoffset, u32 yoffset, u32 zoffset, u32 numPagesX, u32 numPagesY, bool commit) const {
            glTexturePageCommitmentEXT(
                texture_,
                0,
                xoffset * DEFAULT_VIRTUAL_PAGE_SIZE_XYZ,
                yoffset * DEFAULT_VIRTUAL_PAGE_SIZE_XYZ,
                zoffset,
                numPagesX * DEFAULT_VIRTUAL_PAGE_SIZE_XYZ,
                numPagesY * DEFAULT_VIRTUAL_PAGE_SIZE_XYZ,
                1,
                commit ? GL_TRUE : GL_FALSE
            );
        }

    public:
        void bind(i32 activeTexture) const {
            bindAliased(config_.type, activeTexture);
        }

        void bindAliased(TextureType type, i32 activeTexture) const {
            //unbind(activeTexture);
            glActiveTexture(GL_TEXTURE0 + activeTexture);
            glBindTexture(_convertTexture(type), texture_);
        }

        void bindAsImageTexture(u32 unit, i32 mipLevel, bool layered, int32_t layer, ImageTextureAccessMode access) const {
            bindAsImageTexture(unit, mipLevel, layered, layer, access, TextureAccess{ config_.format, config_.storage, config_.dataType });
        }

        void bindAsImageTexture(u32 unit, i32 mipLevel, bool layered, int32_t layer, ImageTextureAccessMode access, const TextureAccess& config) const {
            GLenum accessMode = _convertImageAccessMode(access);
            glBindImageTexture(
                unit,
                texture_,
                mipLevel,
                layered ? GL_TRUE : GL_FALSE,
                layer,
                accessMode,
                _convertInternalFormatPrecise(config.format, config.storage, config.dataType)
            );
        }

        void unbind(i32 activeTexture) const {
            if (activeTexture == -1) return;
            glActiveTexture(GL_TEXTURE0 + activeTexture);
            glBindTexture(_convertTexture(config_.type), 0);
        }

        std::shared_ptr<TextureImpl> copy(const TextureImpl& other) {
            return nullptr;
        }

        const TextureConfig& getConfig() const {
            return config_;
        }

    private:
        static GLenum _convertImageAccessMode(ImageTextureAccessMode access) {
            switch (access) {
            case ImageTextureAccessMode::IMAGE_READ_ONLY: return GL_READ_ONLY;
            case ImageTextureAccessMode::IMAGE_WRITE_ONLY: return GL_WRITE_ONLY;
            case ImageTextureAccessMode::IMAGE_READ_WRITE: return GL_READ_WRITE;
            default: throw std::runtime_error("Unknown image access mode");
            }
        }

        static GLenum _convertTexture(TextureType type) {
            switch (type) {
            case TextureType::TEXTURE_2D:  return GL_TEXTURE_2D;
            case TextureType::TEXTURE_2D_ARRAY: return GL_TEXTURE_2D_ARRAY;
            case TextureType::TEXTURE_CUBE_MAP: return GL_TEXTURE_CUBE_MAP;
            case TextureType::TEXTURE_RECTANGLE: return GL_TEXTURE_RECTANGLE;
            case TextureType::TEXTURE_CUBE_MAP_ARRAY: return GL_TEXTURE_CUBE_MAP_ARRAY;
            case TextureType::TEXTURE_3D: return GL_TEXTURE_3D;
            default: throw std::runtime_error("Unknown texture type");
            }
        }

        // For more information about pixel formats, see https://www.khronos.org/opengl/wiki/Pixel_Transfer#Pixel_format
        static GLenum _convertFormat(TextureComponentFormat format, TextureComponentType type) {
            if (type != TextureComponentType::INT && type != TextureComponentType::UINT) {
                switch (format) {
                case TextureComponentFormat::RED: return GL_RED;
                case TextureComponentFormat::RG: return GL_RG;
                case TextureComponentFormat::RGB: return GL_RGB;
                case TextureComponentFormat::SRGB: return GL_RGB; // GL_RGB even for srgb
                case TextureComponentFormat::RGBA: return GL_RGBA;
                case TextureComponentFormat::SRGB_ALPHA: return GL_RGBA; // GL_RGBA even for srgb_alpha
                case TextureComponentFormat::DEPTH: return GL_DEPTH_COMPONENT;
                case TextureComponentFormat::DEPTH_STENCIL: return GL_DEPTH_STENCIL;
                default: throw std::runtime_error("Unknown format");
                }
            }
            else {
                switch (format) {
                case TextureComponentFormat::RED: return GL_RED_INTEGER;
                case TextureComponentFormat::RG: return GL_RG_INTEGER;
                case TextureComponentFormat::RGB: return GL_RGB_INTEGER;
                case TextureComponentFormat::RGBA: return GL_RGBA_INTEGER;
                default: throw std::runtime_error("Unknown format");
                }
            }
        }

        // See https://gamedev.stackexchange.com/questions/168241/is-gl-depth-component32-deprecated-in-opengl-4-5 for more info on depth component
        static GLint _convertInternalFormat(TextureComponentFormat format, TextureComponentSize size, TextureComponentType type) {
            // If the bits are default we just mirror the format for the internal format option
            if (format == TextureComponentFormat::DEPTH_STENCIL ||
                size == TextureComponentSize::BITS_DEFAULT) {

                switch (format) {
                case TextureComponentFormat::RED: return GL_RED;
                case TextureComponentFormat::RG: return GL_RG;
                case TextureComponentFormat::RGB: return GL_RGB;
                case TextureComponentFormat::SRGB: return GL_SRGB; // Here we specify SRGB since it's internal format
                case TextureComponentFormat::RGBA: return GL_RGBA;
                case TextureComponentFormat::SRGB_ALPHA: return GL_SRGB_ALPHA; // GL_SRGB_ALPHA since it's internal format
                case TextureComponentFormat::DEPTH: return GL_DEPTH_COMPONENT; //GL_DEPTH_COMPONENT;
                case TextureComponentFormat::DEPTH_STENCIL: return GL_DEPTH_STENCIL;
                default: throw std::runtime_error("Unknown format");
                }
            }

            // We don't support specifying bits type other than BITS_DEFAULT for SRGB and SRGB_ALPHA
            if (format == TextureComponentFormat::SRGB || format == TextureComponentFormat::SRGB_ALPHA) {
                throw std::runtime_error("SRGB | SRGB_ALPHA cannot be used without BITS_DEFAULT");
            }

            if (size == TextureComponentSize::BITS_8) {
                if (type == TextureComponentType::INT) {
                    switch (format) {
                    case TextureComponentFormat::RED: return GL_R8I;
                    case TextureComponentFormat::RG: return GL_RG8I;
                    case TextureComponentFormat::RGB: return GL_RGB8I;
                    case TextureComponentFormat::RGBA: return GL_RGBA8I;
                    default: throw std::runtime_error("Unknown combination");
                    }
                }
                if (type == TextureComponentType::UINT) {
                    switch (format) {
                    case TextureComponentFormat::RED: return GL_R8UI;
                    case TextureComponentFormat::RG: return GL_RG8UI;
                    case TextureComponentFormat::RGB: return GL_RGB8UI;
                    case TextureComponentFormat::RGBA: return GL_RGBA8UI;
                    default: throw std::runtime_error("Unknown combination");
                    }
                }
                if (type == TextureComponentType::FLOAT) {
                    switch (format) {
                    case TextureComponentFormat::RED: return GL_R8;
                    case TextureComponentFormat::RG: return GL_RG8;
                    case TextureComponentFormat::RGB: return GL_RGB8;
                    case TextureComponentFormat::RGBA: return GL_RGBA8;
                    default: throw std::runtime_error("Unknown combination");
                    }
                }
            }

            if (size == TextureComponentSize::BITS_16) {
                if (type == TextureComponentType::INT) {
                    switch (format) {
                    case TextureComponentFormat::RED: return GL_R16I;
                    case TextureComponentFormat::RG: return GL_RG16I;
                    case TextureComponentFormat::RGB: return GL_RGB16I;
                    case TextureComponentFormat::RGBA: return GL_RGBA16I;
                    default: throw std::runtime_error("Unknown combination");
                    }
                }
                if (type == TextureComponentType::UINT) {
                    switch (format) {
                    case TextureComponentFormat::RED: return GL_R16UI;
                    case TextureComponentFormat::RG: return GL_RG16UI;
                    case TextureComponentFormat::RGB: return GL_RGB16UI;
                    case TextureComponentFormat::RGBA: return GL_RGBA16UI;
                    default: throw std::runtime_error("Unknown combination");
                    }
                }
                if (type == TextureComponentType::FLOAT) {
                    switch (format) {
                    case TextureComponentFormat::RED: return GL_R16F;
                    case TextureComponentFormat::RG: return GL_RG16F;
                    case TextureComponentFormat::RGB: return GL_RGB16F;
                    case TextureComponentFormat::RGBA: return GL_RGBA16F;
                    case TextureComponentFormat::DEPTH: return GL_DEPTH_COMPONENT16;
                    default: throw std::runtime_error("Unknown combination");
                    }
                }
            }

            if (size == TextureComponentSize::BITS_32) {
                if (type == TextureComponentType::INT) {
                    switch (format) {
                    case TextureComponentFormat::RED: return GL_R32I;
                    case TextureComponentFormat::RG: return GL_RG32I;
                    case TextureComponentFormat::RGB: return GL_RGB32I;
                    case TextureComponentFormat::RGBA: return GL_RGBA32I;
                    default: throw std::runtime_error("Unknown combination");
                    }
                }
                if (type == TextureComponentType::UINT) {
                    switch (format) {
                    case TextureComponentFormat::RED: return GL_R32UI;
                    case TextureComponentFormat::RG: return GL_RG32UI;
                    case TextureComponentFormat::RGB: return GL_RGB32UI;
                    case TextureComponentFormat::RGBA: return GL_RGBA32UI;
                    default: throw std::runtime_error("Unknown combination");
                    }
                }
                if (type == TextureComponentType::FLOAT) {
                    switch (format) {
                    case TextureComponentFormat::RED: return GL_R32F;
                    case TextureComponentFormat::RG: return GL_RG32F;
                    case TextureComponentFormat::RGB: return GL_RGB32F;
                    case TextureComponentFormat::RGBA: return GL_RGBA32F;
                    case TextureComponentFormat::DEPTH: return GL_DEPTH_COMPONENT32F;
                    default: throw std::runtime_error("Unknown combination");
                    }
                }
            }

            if (size == TextureComponentSize::BITS_11_11_10) {
                if (type == TextureComponentType::FLOAT) {
                    if (format == TextureComponentFormat::RGB) {
                        return GL_R11F_G11F_B10F;
                    }
                    else {
                        throw std::runtime_error("Invalid 11_11_10 combination");
                    }
                }
                else {
                    throw std::runtime_error("Unable to use types other than float for 11_11_10");
                }
            }

            throw std::runtime_error("Unknown error occurred");
        }

        static GLint _convertInternalFormatPrecise(TextureComponentFormat format, TextureComponentSize size, TextureComponentType type) {
            // If the bits are default we just mirror the format for the internal format option
            if (format == TextureComponentFormat::DEPTH_STENCIL ||
                size == TextureComponentSize::BITS_DEFAULT) {

                switch (format) {
                case TextureComponentFormat::RED: return GL_R8;
                case TextureComponentFormat::RG: return GL_RG8;
                case TextureComponentFormat::RGB: return GL_RGB8;
                case TextureComponentFormat::SRGB: return GL_SRGB8; // Here we specify SRGB since it's internal format
                case TextureComponentFormat::RGBA: return GL_RGBA8;
                case TextureComponentFormat::SRGB_ALPHA: return GL_SRGB8_ALPHA8; // GL_SRGB_ALPHA since it's internal format
                case TextureComponentFormat::DEPTH: return GL_DEPTH_COMPONENT24; //GL_DEPTH_COMPONENT;
                case TextureComponentFormat::DEPTH_STENCIL: return GL_DEPTH24_STENCIL8;
                default: throw std::runtime_error("Unknown format");
                }
            }

            // We don't support specifying bits type other than BITS_DEFAULT for SRGB and SRGB_ALPHA
            if (format == TextureComponentFormat::SRGB || format == TextureComponentFormat::SRGB_ALPHA) {
                throw std::runtime_error("SRGB | SRGB_ALPHA cannot be used without BITS_DEFAULT");
            }

            if (size == TextureComponentSize::BITS_8) {
                if (type == TextureComponentType::INT) {
                    switch (format) {
                    case TextureComponentFormat::RED: return GL_R8I;
                    case TextureComponentFormat::RG: return GL_RG8I;
                    case TextureComponentFormat::RGB: return GL_RGB8I;
                    case TextureComponentFormat::RGBA: return GL_RGBA8I;
                    default: throw std::runtime_error("Unknown combination");
                    }
                }
                if (type == TextureComponentType::UINT) {
                    switch (format) {
                    case TextureComponentFormat::RED: return GL_R8UI;
                    case TextureComponentFormat::RG: return GL_RG8UI;
                    case TextureComponentFormat::RGB: return GL_RGB8UI;
                    case TextureComponentFormat::RGBA: return GL_RGBA8UI;
                    default: throw std::runtime_error("Unknown combination");
                    }
                }
                if (type == TextureComponentType::FLOAT) {
                    switch (format) {
                    case TextureComponentFormat::RED: return GL_R8;
                    case TextureComponentFormat::RG: return GL_RG8;
                    case TextureComponentFormat::RGB: return GL_RGB8;
                    case TextureComponentFormat::RGBA: return GL_RGBA8;
                    default: throw std::runtime_error("Unknown combination");
                    }
                }
            }

            if (size == TextureComponentSize::BITS_16) {
                if (type == TextureComponentType::INT) {
                    switch (format) {
                    case TextureComponentFormat::RED: return GL_R16I;
                    case TextureComponentFormat::RG: return GL_RG16I;
                    case TextureComponentFormat::RGB: return GL_RGB16I;
                    case TextureComponentFormat::RGBA: return GL_RGBA16I;
                    default: throw std::runtime_error("Unknown combination");
                    }
                }
                if (type == TextureComponentType::UINT) {
                    switch (format) {
                    case TextureComponentFormat::RED: return GL_R16UI;
                    case TextureComponentFormat::RG: return GL_RG16UI;
                    case TextureComponentFormat::RGB: return GL_RGB16UI;
                    case TextureComponentFormat::RGBA: return GL_RGBA16UI;
                    default: throw std::runtime_error("Unknown combination");
                    }
                }
                if (type == TextureComponentType::FLOAT) {
                    switch (format) {
                    case TextureComponentFormat::RED: return GL_R16F;
                    case TextureComponentFormat::RG: return GL_RG16F;
                    case TextureComponentFormat::RGB: return GL_RGB16F;
                    case TextureComponentFormat::RGBA: return GL_RGBA16F;
                    case TextureComponentFormat::DEPTH: return GL_DEPTH_COMPONENT16;
                    default: throw std::runtime_error("Unknown combination");
                    }
                }
            }

            if (size == TextureComponentSize::BITS_32) {
                if (type == TextureComponentType::INT) {
                    switch (format) {
                    case TextureComponentFormat::RED: return GL_R32I;
                    case TextureComponentFormat::RG: return GL_RG32I;
                    case TextureComponentFormat::RGB: return GL_RGB32I;
                    case TextureComponentFormat::RGBA: return GL_RGBA32I;
                    default: throw std::runtime_error("Unknown combination");
                    }
                }
                if (type == TextureComponentType::UINT) {
                    switch (format) {
                    case TextureComponentFormat::RED: return GL_R32UI;
                    case TextureComponentFormat::RG: return GL_RG32UI;
                    case TextureComponentFormat::RGB: return GL_RGB32UI;
                    case TextureComponentFormat::RGBA: return GL_RGBA32UI;
                    default: throw std::runtime_error("Unknown combination");
                    }
                }
                if (type == TextureComponentType::FLOAT) {
                    switch (format) {
                    case TextureComponentFormat::RED: return GL_R32F;
                    case TextureComponentFormat::RG: return GL_RG32F;
                    case TextureComponentFormat::RGB: return GL_RGB32F;
                    case TextureComponentFormat::RGBA: return GL_RGBA32F;
                    case TextureComponentFormat::DEPTH: return GL_DEPTH_COMPONENT32F;
                    default: throw std::runtime_error("Unknown combination");
                    }
                }
            }

            if (size == TextureComponentSize::BITS_11_11_10) {
                if (type == TextureComponentType::FLOAT) {
                    if (format == TextureComponentFormat::RGB) {
                        return GL_R11F_G11F_B10F;
                    }
                    else {
                        throw std::runtime_error("Invalid 11_11_10 combination");
                    }
                }
                else {
                    throw std::runtime_error("Unable to use types other than float for 11_11_10");
                }
            }

            throw std::runtime_error("Unknown error occurred");
        }

        static GLenum _convertType(TextureComponentType type, TextureComponentSize size) {
            if (size == TextureComponentSize::BITS_DEFAULT || size == TextureComponentSize::BITS_8) {
                if (type == TextureComponentType::INT || type == TextureComponentType::INT_NORM) return GL_BYTE;
                if (type == TextureComponentType::UINT || type == TextureComponentType::UINT_NORM) return GL_UNSIGNED_BYTE;
                if (type == TextureComponentType::FLOAT) return GL_FLOAT;
            }

            if (size == TextureComponentSize::BITS_16) {
                if (type == TextureComponentType::INT) return GL_SHORT;
                if (type == TextureComponentType::UINT) return GL_UNSIGNED_SHORT;
                if (type == TextureComponentType::FLOAT) return GL_FLOAT; // TODO: GL_HALF_FLOAT?
            }

            if (size == TextureComponentSize::BITS_32) {
                if (type == TextureComponentType::INT) return GL_INT;
                if (type == TextureComponentType::UINT) return GL_UNSIGNED_INT;
                if (type == TextureComponentType::FLOAT) return GL_FLOAT;
            }

            if (size == TextureComponentSize::BITS_11_11_10) {
                if (type == TextureComponentType::FLOAT) return GL_FLOAT;
            }

            throw std::runtime_error("Unknown combination");
        }

        static GLint _convertTextureCoordinateWrapping(TextureCoordinateWrapping wrap) {
            switch (wrap) {
            case TextureCoordinateWrapping::REPEAT: return GL_REPEAT;
            case TextureCoordinateWrapping::NEAREST: return GL_NEAREST;
            case TextureCoordinateWrapping::LINEAR: return GL_LINEAR;
            case TextureCoordinateWrapping::MIRRORED_REPEAT: return GL_MIRRORED_REPEAT;
            case TextureCoordinateWrapping::CLAMP_TO_EDGE: return GL_CLAMP_TO_EDGE;
            case TextureCoordinateWrapping::CLAMP_TO_BORDER: return GL_CLAMP_TO_BORDER;
            case TextureCoordinateWrapping::MIRROR_CLAMP_TO_EDGE: return GL_MIRROR_CLAMP_TO_EDGE;
            default: throw std::runtime_error("Unknown coordinate wrapping");
            }
        }

        static GLint _convertTextureMinFilter(TextureMinificationFilter min) {
            switch (min) {
            case TextureMinificationFilter::NEAREST: return GL_NEAREST;
            case TextureMinificationFilter::LINEAR: return GL_LINEAR;
            case TextureMinificationFilter::NEAREST_MIPMAP_NEAREST: return GL_NEAREST_MIPMAP_NEAREST;
            case TextureMinificationFilter::LINEAR_MIPMAP_NEAREST: return GL_LINEAR_MIPMAP_NEAREST;
            case TextureMinificationFilter::NEAREST_MIPMAP_LINEAR: return GL_NEAREST_MIPMAP_LINEAR;
            case TextureMinificationFilter::LINEAR_MIPMAP_LINEAR: return GL_LINEAR_MIPMAP_LINEAR;
            default: throw std::runtime_error("Unknown min filter");
            }
        }

        static GLint _convertTextureMagFilter(TextureMagnificationFilter mag) {
            switch (mag) {
            case TextureMagnificationFilter::NEAREST: return GL_NEAREST;
            case TextureMagnificationFilter::LINEAR: return GL_LINEAR;
            default: throw std::runtime_error("Unknown mag filter");
            }
        }

        static GLenum _convertTextureCompareMode(TextureCompareMode mode) {
            switch (mode) {
            case TextureCompareMode::NONE: return GL_NONE;
            case TextureCompareMode::COMPARE_REF_TO_TEXTURE: return GL_COMPARE_REF_TO_TEXTURE;
            default: throw std::runtime_error("Unknown compare mode");
            }
        }

        static GLenum _convertTextureCompareFunc(TextureCompareFunc func) {
            switch (func) {
            case TextureCompareFunc::ALWAYS: return GL_ALWAYS;
            case TextureCompareFunc::NEVER: return GL_NEVER;
            case TextureCompareFunc::EQUAL: return GL_EQUAL;
            case TextureCompareFunc::NOTEQUAL: return GL_NOTEQUAL;
            case TextureCompareFunc::LESS: return GL_LESS;
            case TextureCompareFunc::LEQUAL: return GL_LEQUAL;
            case TextureCompareFunc::GREATER: return GL_GREATER;
            case TextureCompareFunc::GEQUAL: return GL_GEQUAL;
            default: throw std::runtime_error("Unknown compare func");
            }
        }
    };

    Texture::Texture() {}
    Texture::Texture(const TextureConfig& config, const TextureArrayData& data, bool initHandle) {
        impl_ = std::make_shared<TextureImpl>(config, data, initHandle);
    }

    Texture::~Texture() {}

    void Texture::SetCoordinateWrapping(TextureCoordinateWrapping wrap) { EnsureValid_(); impl_->setCoordinateWrapping(wrap); }
    void Texture::SetMinMagFilter(TextureMinificationFilter min, TextureMagnificationFilter mag) { EnsureValid_(); impl_->setMinMagFilter(min, mag); }
    void Texture::SetTextureCompare(TextureCompareMode mode, TextureCompareFunc func) { EnsureValid_(); impl_->setTextureCompare(mode, func); }

    TextureType Texture::Type() const { EnsureValid_(); return impl_->type(); }
    TextureComponentFormat Texture::Format() const { EnsureValid_(); return impl_->format(); }
    TextureHandle Texture::Handle() const { EnsureValid_(); return impl_->handle(); }

    GpuTextureHandle Texture::GpuHandle() const { EnsureValid_(); return impl_->GpuHandle(); }

    void Texture::MakeResident_(const Texture& texture) { TextureImpl::MakeResident(texture); }
    void Texture::MakeNonResident_(const Texture& texture) { TextureImpl::MakeNonResident(texture); }

    u32 Texture::Width() const { EnsureValid_(); return impl_->width(); }
    u32 Texture::Height() const { EnsureValid_(); return impl_->height(); }
    u32 Texture::Depth() const { EnsureValid_(); return impl_->depth(); }

    void Texture::Bind(i32 activeTexture) const { EnsureValid_(); impl_->bind(activeTexture); }
    void Texture::BindAliased(TextureType type, i32 activeTexture) const { EnsureValid_(); impl_->bindAliased(type, activeTexture); }
    void Texture::BindAsImageTexture(u32 unit, i32 mipLevel, bool layered, int32_t layer, ImageTextureAccessMode access) const {
        EnsureValid_(); impl_->bindAsImageTexture(unit, mipLevel, layered, layer, access);
    }
    void Texture::BindAsImageTexture(u32 unit, i32 mipLevel, bool layered, int32_t layer, ImageTextureAccessMode access, const TextureAccess& config) const {
        EnsureValid_(); impl_->bindAsImageTexture(unit, mipLevel, layered, layer, access, config);
    }

    void Texture::Unbind(i32 activeTexture) const { EnsureValid_(); impl_->unbind(activeTexture); }
    bool Texture::Valid() const { return impl_ != nullptr; }

    void Texture::Clear(const i32 mipLevel, const void* clearValue) const { EnsureValid_(); impl_->Clear(mipLevel, clearValue); }
    void Texture::ClearLayer(const i32 mipLevel, const i32 layer, const void* clearValue) const { EnsureValid_(); impl_->clearLayer(mipLevel, layer, clearValue); }
    void Texture::ClearLayerRegion(
        const i32 mipLevel,
        const i32 layer,
        const i32 xoffset,
        const i32 yoffset,
        const i32 width,
        const i32 height,
        const void* clearValue) const {

        EnsureValid_();
        impl_->ClearLayerRegion(mipLevel, layer, xoffset, yoffset, width, height, clearValue);
    }

    const void* Texture::Underlying() const { EnsureValid_(); return impl_->Underlying(); }

    u32 Texture::VirtualPageSizeXY() {
        return DEFAULT_VIRTUAL_PAGE_SIZE_XYZ;
    }

    void Texture::CommitOrUncommitVirtualPage(u32 xoffset, u32 yoffset, u32 zoffset, u32 numPagesX, u32 numPagesY, bool commit) const {
        EnsureValid_();
        impl_->CommitOrUncommitVirtualPage(xoffset, yoffset, zoffset, numPagesX, numPagesY, commit);
    }

    size_t Texture::HashCode() const {
        return std::hash<void*>{}((void*)impl_.get());
    }

    bool Texture::operator==(const Texture& other) const {
        return impl_ == other.impl_;
    }

    bool Texture::operator!=(const Texture& other) const {
        return !(this->operator==(other));
    }

    // Creates a new texture and copies this texture into it
    Texture Texture::Copy(u32 newWidth, u32 newHeight) const {
        throw std::runtime_error("Must implement");
    }

    const TextureConfig& Texture::GetConfig() const {
        EnsureValid_(); return impl_->getConfig();
    }

    void Texture::SetHandle_(const TextureHandle handle) {
        EnsureValid_(); impl_->setHandle(handle);
    }

    void Texture::EnsureValid_() const {
        if (impl_ == nullptr) {
            throw std::runtime_error("Attempt to use null texture");
        }
    }

    TextureMemResidencyGuard::TextureMemResidencyGuard()
        : TextureMemResidencyGuard(Texture()) {}

    TextureMemResidencyGuard::TextureMemResidencyGuard(const Texture& texture)
        : texture_(texture) {

        IncrementRefcount_();
    }

    TextureMemResidencyGuard::TextureMemResidencyGuard(TextureMemResidencyGuard&& other) noexcept {
        this->operator=(other);
    }

    TextureMemResidencyGuard::TextureMemResidencyGuard(const TextureMemResidencyGuard& other) noexcept {
        this->operator=(other);
    }

    TextureMemResidencyGuard& TextureMemResidencyGuard::operator=(TextureMemResidencyGuard&& other) noexcept {
        Copy_(other);
        return *this;
    }

    TextureMemResidencyGuard& TextureMemResidencyGuard::operator=(const TextureMemResidencyGuard& other) noexcept {
        Copy_(other);
        return *this;
    }

    void TextureMemResidencyGuard::Copy_(const TextureMemResidencyGuard& other) {
        if (this->texture_ == other.texture_) return;

        DecrementRefcount_();
        this->texture_ = other.texture_;
        IncrementRefcount_();
    }

    void TextureMemResidencyGuard::IncrementRefcount_() {
        if (texture_ == Texture()) return;

        texture_.impl_->memRefcount_ += 1;
        if (texture_.impl_->memRefcount_ == 1) {
            Texture::MakeResident_(texture_);
        }
    }

    void TextureMemResidencyGuard::DecrementRefcount_() {
        if (texture_ == Texture()) return;

        texture_.impl_->memRefcount_ -= 1;
        if (texture_.impl_->memRefcount_ == 0) {
            Texture::MakeNonResident_(texture_);
        }
    }

    TextureMemResidencyGuard::~TextureMemResidencyGuard() {
        DecrementRefcount_();
    }
}