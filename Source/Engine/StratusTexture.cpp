#include "StratusTexture.h"
#include "StratusLog.h"
#include <GL/gl3w.h>
#include <exception>
#include <unordered_set>
#include <iostream>
#include "StratusApplicationThread.h"
#include "StratusGraphicsDriver.h"

namespace stratus {
    static const void * CastTexDataToPtr(const TextureArrayData& data, const size_t offset) {
        if (!data.size()) return nullptr;
        return data[offset].data;
    }

    class TextureImpl {
        GLuint _texture;
        TextureConfig _config;
        mutable int _activeTexture = -1;
        TextureHandle handle_;

    public:
        TextureImpl(const TextureConfig & config, const TextureArrayData& data, bool initHandle) {
            if (initHandle) {
                handle_ = TextureHandle::NextHandle();
            }

            glGenTextures(1, &_texture);

            _config = config;

            bind();
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
                    _convertFormat(config.format), // format (e.g. RGBA)
                    _convertType(config.dataType, config.storage), // type (e.g. FLOAT)
                    CastTexDataToPtr(data, 0)
                );

                // Set anisotropic filtering
                auto maxAnisotropy = GraphicsDriver::GetConfig().maxAnisotropy;
                //maxAnisotropy = maxAnisotropy > 2.0f ? 2.0f : maxAnisotropy;
                glTexParameterf(_convertTexture(config.type), GL_TEXTURE_MAX_ANISOTROPY, maxAnisotropy);
            }
            else if (config.type == TextureType::TEXTURE_2D_ARRAY) {
                // See: https://johanmedestrom.wordpress.com/2016/03/18/opengl-cascaded-shadow-maps/
                // for an example of glTexImage3D
                glTexImage3D(GL_TEXTURE_2D_ARRAY, // target
                    0, // level 
                    _convertInternalFormat(config.format, config.storage, config.dataType), // internal format (e.g. RGBA16F)
                    config.width, 
                    config.height, 
                    config.depth,
                    0,
                    _convertFormat(config.format), // format (e.g. RGBA)
                    _convertType(config.dataType, config.storage), // type (e.g. FLOAT)
                    CastTexDataToPtr(data, 0)
                );
            }
            else if (config.type == TextureType::TEXTURE_CUBE_MAP) {
                if (config.width != config.height || (config.depth % 6) != 0) {
                    throw std::runtime_error("Unable to create cube map texture - invalid width/height or depth");
                }

                for (int face = 0; face < 6; ++face) {
                    glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + face, 
                        0, 
                        _convertInternalFormat(config.format, config.storage, config.dataType),
                        config.width, 
                        config.height,
                        0, 
                        _convertFormat(config.format),
                        _convertType(config.dataType, config.storage), 
                        CastTexDataToPtr(data, (const size_t)face)
                    );
                }
            }
            else {
                throw std::runtime_error("Unknown texture type specified");
            }

            unbind();

            // Mipmaps aren't generated for rectangle textures
            if (config.generateMipMaps && config.type != TextureType::TEXTURE_RECTANGLE) glGenerateTextureMipmap(_texture);
        }

        ~TextureImpl() {
            if (ApplicationThread::Instance()->CurrentIsApplicationThread()) {
                glDeleteTextures(1, &_texture);
            }
            else {
                auto texture = _texture;
                ApplicationThread::Instance()->Queue([texture]() { GLuint tex = texture; glDeleteTextures(1, &tex); });
            }
        }

        // No copying
        TextureImpl(const TextureImpl &) = delete;
        TextureImpl(TextureImpl &&) = delete;
        TextureImpl & operator=(const TextureImpl &) = delete;
        TextureImpl & operator=(TextureImpl &&) = delete;

        void setCoordinateWrapping(TextureCoordinateWrapping wrap) {
            if (_config.type == TextureType::TEXTURE_RECTANGLE && (wrap != TextureCoordinateWrapping::CLAMP_TO_BORDER && wrap != TextureCoordinateWrapping::CLAMP_TO_EDGE)) {
                STRATUS_ERROR << "Texture_Rectangle ONLY supports clamp to edge and clamp to border" << std::endl;
                throw std::runtime_error("Invalid Texture_Rectangle coordinate wrapping");
            }

            bind();
            glTexParameteri(_convertTexture(_config.type), GL_TEXTURE_WRAP_S, _convertTextureCoordinateWrapping(wrap));
            glTexParameteri(_convertTexture(_config.type), GL_TEXTURE_WRAP_T, _convertTextureCoordinateWrapping(wrap));
            // Support third dimension for cube maps
            if (_config.type == TextureType::TEXTURE_CUBE_MAP) glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, _convertTextureCoordinateWrapping(wrap));
            unbind();
        }

        void setMinMagFilter(TextureMinificationFilter min, TextureMagnificationFilter mag) {
            bind();
            glTexParameteri(_convertTexture(_config.type), GL_TEXTURE_MIN_FILTER, _convertTextureMinFilter(min));
            glTexParameteri(_convertTexture(_config.type), GL_TEXTURE_MAG_FILTER, _convertTextureMagFilter(mag));
            unbind();
        }

        void setTextureCompare(TextureCompareMode mode, TextureCompareFunc func) {
            bind();
            glTexParameteri(_convertTexture(_config.type), GL_TEXTURE_COMPARE_MODE, _convertTextureCompareMode(mode));
            glTexParameterf(_convertTexture(_config.type), GL_TEXTURE_COMPARE_FUNC, _convertTextureCompareFunc(func));
            unbind();
        }

        void setHandle(const TextureHandle handle) {
            handle_ = handle;
        }

        void Clear(const int mipLevel, const void * clearValue) {
            glClearTexImage(_texture, mipLevel,
                _convertFormat(_config.format), // format (e.g. RGBA)
                _convertType(_config.dataType, _config.storage), // type (e.g. FLOAT))
                clearValue); 
        }

        // See https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glClearTexSubImage.xhtml
        // for information about how to handle different texture types.
        //
        // This does not work for compressed textures or texture buffers.
        void clearLayer(const int mipLevel, const int layer, const void * clearValue) {
            if (type() == TextureType::TEXTURE_2D || type() == TextureType::TEXTURE_RECTANGLE) {
                Clear(mipLevel, clearValue);
            }
            else {
                const int xoffset = 0, yoffset = 0;
                const int zoffset = layer;
                const int depth = 1; // number of layers to clear
                glClearTexSubImage(_texture, mipLevel, xoffset, yoffset, zoffset, width(), height(), depth,
                    _convertFormat(_config.format), // format (e.g. RGBA)
                    _convertType(_config.dataType, _config.storage), // type (e.g. FLOAT))
                    clearValue);
            }
        }

        TextureType type() const               { return _config.type; }
        TextureComponentFormat format() const  { return _config.format; }
        TextureHandle handle() const           { return handle_; }

        // These cause RenderDoc to disable frame capture... super unfortunate
        GpuTextureHandle GpuHandle() const {
            auto gpuHandle = glGetTextureHandleARB(_texture);
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

        uint32_t width() const                 { return _config.width; }
        uint32_t height() const                { return _config.height; }
        uint32_t depth() const                 { return _config.depth; }
        void * Underlying() const              { return (void *)&_texture; }

    public:
        void bind(int activeTexture = 0) const {
            unbind();
            glActiveTexture(GL_TEXTURE0 + activeTexture);
            glBindTexture(_convertTexture(_config.type), _texture);
            _activeTexture = activeTexture;
        }

        void bindAsImageTexture(uint32_t unit, bool layered, int32_t layer, ImageTextureAccessMode access) const {
            GLenum accessMode = _convertImageAccessMode(access);
            glBindImageTexture(unit, 
                               _texture, 
                               0, 
                               layered ? GL_TRUE : GL_FALSE,
                               layer,
                               accessMode,
                               _convertInternalFormat(_config.format, _config.storage, _config.dataType));
        }

        void unbind() const {
            if (_activeTexture == -1) return;
            glActiveTexture(GL_TEXTURE0 + _activeTexture);
            glBindTexture(_convertTexture(_config.type), 0);
            _activeTexture = -1;
        }

        std::shared_ptr<TextureImpl> copy(const TextureImpl & other) {
            return nullptr;
        }

        const TextureConfig & getConfig() const {
            return _config;
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
            default: throw std::runtime_error("Unknown texture type");
            }
        }

        static GLenum _convertFormat(TextureComponentFormat format) {
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

        // See https://gamedev.stackexchange.com/questions/168241/is-gl-depth-component32-deprecated-in-opengl-4-5 for more info on depth component
        static GLint _convertInternalFormat(TextureComponentFormat format, TextureComponentSize size, TextureComponentType type) {
            // If the bits are default we just mirror the format for the internal format option
            if (format == TextureComponentFormat::DEPTH_STENCIL || size == TextureComponentSize::BITS_DEFAULT) {
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

        static GLenum _convertType(TextureComponentType type, TextureComponentSize size) {
            if (size == TextureComponentSize::BITS_DEFAULT || size == TextureComponentSize::BITS_8) {
                if (type == TextureComponentType::INT) return GL_BYTE;
                if (type == TextureComponentType::UINT) return GL_UNSIGNED_BYTE;
                if (type == TextureComponentType::FLOAT) return GL_FLOAT;
            }

            if (size == TextureComponentSize::BITS_16) {
                if (type == TextureComponentType::INT) return GL_SHORT;
                if (type == TextureComponentType::UINT) return GL_UNSIGNED_SHORT;
                if (type == TextureComponentType::FLOAT) return GL_FLOAT; // TODO: GL_HALF_FLOAT?
            }

            if (size == TextureComponentSize::BITS_32) {
                if (type == TextureComponentType::INT) return GL_SHORT;
                if (type == TextureComponentType::UINT) return GL_UNSIGNED_SHORT;
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
    Texture::Texture(const TextureConfig & config, const TextureArrayData& data, bool initHandle) {
        impl_ = std::make_shared<TextureImpl>(config, data, initHandle);
    }

    Texture::~Texture() {}

    void Texture::SetCoordinateWrapping(TextureCoordinateWrapping wrap) { impl_->setCoordinateWrapping(wrap); }
    void Texture::SetMinMagFilter(TextureMinificationFilter min, TextureMagnificationFilter mag) { impl_->setMinMagFilter(min, mag); }
    void Texture::SetTextureCompare(TextureCompareMode mode, TextureCompareFunc func) { impl_->setTextureCompare(mode, func); }

    TextureType Texture::Type() const { return impl_->type(); }
    TextureComponentFormat Texture::Format() const { return impl_->format(); }
    TextureHandle Texture::Handle() const { return impl_->handle(); }

    GpuTextureHandle Texture::GpuHandle() const { return impl_->GpuHandle(); }

    void Texture::MakeResident(const Texture& texture) { TextureImpl::MakeResident(texture); }
    void Texture::MakeNonResident(const Texture& texture) { TextureImpl::MakeNonResident(texture); }

    uint32_t Texture::Width() const { return impl_->width(); }
    uint32_t Texture::Height() const { return impl_->height(); }
    uint32_t Texture::Depth() const { return impl_->depth(); }

    void Texture::Bind(int activeTexture) const { impl_->bind(activeTexture); }
    void Texture::BindAsImageTexture(uint32_t unit, bool layered, int32_t layer, ImageTextureAccessMode access) const {
        impl_->bindAsImageTexture(unit, layered, layer, access);
    }
    void Texture::Unbind() const { impl_->unbind(); }
    bool Texture::Valid() const { return impl_ != nullptr; }

    void Texture::Clear(const int mipLevel, const void * clearValue) { impl_->Clear(mipLevel, clearValue); }
    void Texture::ClearLayer(const int mipLevel, const int layer, const void * clearValue) { impl_->clearLayer(mipLevel, layer, clearValue); }

    const void * Texture::Underlying() const { return impl_->Underlying(); }

    size_t Texture::HashCode() const {
        return std::hash<void *>{}((void *)impl_.get());
    }

    bool Texture::operator==(const Texture & other) const {
        return impl_ == other.impl_;
    }

    // Creates a new texture and copies this texture into it
    Texture Texture::Copy(uint32_t newWidth, uint32_t newHeight) {
        throw std::runtime_error("Must implement");
    }

    const TextureConfig & Texture::GetConfig() const {
        return impl_->getConfig();
    }

    void Texture::SetHandle_(const TextureHandle handle) {
        impl_->setHandle(handle);
    }
}