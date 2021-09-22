#include "StratusTexture.h"
#include <GL/gl3w.h>
#include <exception>
#include <unordered_set>
#include <iostream>

namespace stratus {
    class TextureImpl {
        GLuint _texture;
        TextureConfig _config;
        mutable int _activeTexture = -1;
        TextureHandle _handle;

    public:
        TextureImpl(const TextureConfig & config, const void * data, bool initHandle) {
            if (initHandle) {
                _handle = TextureHandle::NextHandle();
            }

            glGenTextures(1, &_texture);

            _config = config;

            bind();
            if (config.type == TextureType::TEXTURE_2D) {
                glTexImage2D(GL_TEXTURE_2D, // target
                    0, // level 
                    _convertInternalFormat(config.format, config.storage, config.dataType), // internal format (e.g. RGBA16F)
                    config.width, 
                    config.height,
                    0,
                    _convertFormat(config.format), // format (e.g. RGBA)
                    _convertType(config.dataType, config.storage), // type (e.g. FLOAT)
                    data
                );
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
                    data
                );
            }
            else {
                for (int face = 0; face < 6; ++face) {
                    glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + face, 
                        0, 
                        _convertInternalFormat(config.format, config.storage, config.dataType),
                        config.width, 
                        config.height,
                        0, 
                        _convertFormat(config.format),
                        _convertType(config.dataType, config.storage), 
                        data
                    );
                }
            }
            if (config.generateMipMaps) glGenerateMipmap(_convertTexture(_config.type));
            unbind();
        }

        ~TextureImpl() {
            glDeleteTextures(1, &_texture);
        }

        // No copying
        TextureImpl(const TextureImpl &) = delete;
        TextureImpl(TextureImpl &&) = delete;
        TextureImpl & operator=(const TextureImpl &) = delete;
        TextureImpl & operator=(TextureImpl &&) = delete;

        void setCoordinateWrapping(TextureCoordinateWrapping wrap) {
            bind();
            glTexParameteri(_convertTexture(_config.type), GL_TEXTURE_WRAP_S, _convertTextureCoordinateWrapping(wrap));
            glTexParameteri(_convertTexture(_config.type), GL_TEXTURE_WRAP_T, _convertTextureCoordinateWrapping(wrap));
            // Support third dimension for cube maps
            if (_config.type == TextureType::TEXTURE_3D) glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, _convertTextureCoordinateWrapping(wrap));
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
            _handle = handle;
        }

        TextureType type() const              { return _config.type; }
        TextureComponentFormat format() const { return _config.format; }
        TextureHandle handle() const         { return _handle; }
        uint32_t width() const                { return _config.width; }
        uint32_t height() const               { return _config.height; }
        uint32_t depth() const                { return _config.depth; }
        void * underlying() const             { return (void *)&_texture; }

    public:
        void bind(int activeTexture = 0) const {
            unbind();
            glActiveTexture(GL_TEXTURE0 + activeTexture);
            glBindTexture(_convertTexture(_config.type), _texture);
            _activeTexture = activeTexture;
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
        static GLenum _convertTexture(TextureType type) {
            switch (type) {
            case TextureType::TEXTURE_2D:  return GL_TEXTURE_2D;
            case TextureType::TEXTURE_2D_ARRAY: return GL_TEXTURE_2D_ARRAY;
            case TextureType::TEXTURE_3D: return GL_TEXTURE_CUBE_MAP;
            default: throw std::runtime_error("Unknown texture type");
            }
        }

        static GLenum _convertFormat(TextureComponentFormat format) {
            switch (format) {
                case TextureComponentFormat::RED: return GL_RED;
                case TextureComponentFormat::RGB: return GL_RGB;
                case TextureComponentFormat::SRGB: return GL_RGB; // GL_RGB even for srgb
                case TextureComponentFormat::RGBA: return GL_RGBA;
                case TextureComponentFormat::SRGB_ALPHA: return GL_RGBA; // GL_RGBA even for srgb_alpha
                case TextureComponentFormat::DEPTH: return GL_DEPTH_COMPONENT;
                case TextureComponentFormat::DEPTH_STENCIL: return GL_DEPTH_STENCIL;
                default: throw std::runtime_error("Unknown format");
            }
        }

        static GLint _convertInternalFormat(TextureComponentFormat format, TextureComponentSize size, TextureComponentType type) {
            // If the bits are default we just mirror the format for the internal format option
            if (format == TextureComponentFormat::DEPTH || format == TextureComponentFormat::DEPTH_STENCIL || size == TextureComponentSize::BITS_DEFAULT) {
                switch (format) {
                    case TextureComponentFormat::RED: return GL_RED;
                    case TextureComponentFormat::RGB: return GL_RGB;
                    case TextureComponentFormat::SRGB: return GL_SRGB; // Here we specify SRGB since it's internal format
                    case TextureComponentFormat::RGBA: return GL_RGBA;
                    case TextureComponentFormat::SRGB_ALPHA: return GL_SRGB_ALPHA; // GL_SRGB_ALPHA since it's internal format
                    case TextureComponentFormat::DEPTH: return GL_DEPTH_COMPONENT;
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
                        case TextureComponentFormat::RGB: return GL_RGB8I;
                        case TextureComponentFormat::RGBA: return GL_RGBA8I;
                        default: throw std::runtime_error("Unknown combination");
                    }
                }
                if (type == TextureComponentType::UINT) {
                    switch (format) {
                        case TextureComponentFormat::RED: return GL_R8UI;
                        case TextureComponentFormat::RGB: return GL_RGB8UI;
                        case TextureComponentFormat::RGBA: return GL_RGBA8UI;
                        default: throw std::runtime_error("Unknown combination");
                    }
                }
                if (type == TextureComponentType::FLOAT) {
                    throw std::runtime_error("FLOAT cannot be specified with 8 bit type");
                }
            }

            if (size == TextureComponentSize::BITS_16) {
                if (type == TextureComponentType::INT) {
                    switch (format) {
                        case TextureComponentFormat::RED: return GL_R16I;
                        case TextureComponentFormat::RGB: return GL_RGB16I;
                        case TextureComponentFormat::RGBA: return GL_RGBA16I;
                        default: throw std::runtime_error("Unknown combination");
                    }
                }
                if (type == TextureComponentType::UINT) {
                    switch (format) {
                        case TextureComponentFormat::RED: return GL_R16UI;
                        case TextureComponentFormat::RGB: return GL_RGB16UI;
                        case TextureComponentFormat::RGBA: return GL_RGBA16UI;
                        default: throw std::runtime_error("Unknown combination");
                    }
                }
                if (type == TextureComponentType::FLOAT) {
                    switch (format) {
                        case TextureComponentFormat::RED: return GL_R16F;
                        case TextureComponentFormat::RGB: return GL_RGB16F;
                        case TextureComponentFormat::RGBA: return GL_RGBA16F;
                        default: throw std::runtime_error("Unknown combination");
                    }
                }
            }

            if (size == TextureComponentSize::BITS_32) {
                if (type == TextureComponentType::INT) {
                    switch (format) {
                        case TextureComponentFormat::RED: return GL_R32I;
                        case TextureComponentFormat::RGB: return GL_RGB32I;
                        case TextureComponentFormat::RGBA: return GL_RGBA32I;
                        default: throw std::runtime_error("Unknown combination");
                    }
                }
                if (type == TextureComponentType::UINT) {
                    switch (format) {
                        case TextureComponentFormat::RED: return GL_R32UI;
                        case TextureComponentFormat::RGB: return GL_RGB32UI;
                        case TextureComponentFormat::RGBA: return GL_RGBA32UI;
                        default: throw std::runtime_error("Unknown combination");
                    }
                }
                if (type == TextureComponentType::FLOAT) {
                    switch (format) {
                        case TextureComponentFormat::RED: return GL_R32F;
                        case TextureComponentFormat::RGB: return GL_RGB32F;
                        case TextureComponentFormat::RGBA: return GL_RGBA32F;
                        default: throw std::runtime_error("Unknown combination");
                    }
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

            throw std::runtime_error("Unknown combination");
        }

        static GLint _convertTextureCoordinateWrapping(TextureCoordinateWrapping wrap) {
            switch (wrap) {
                case TextureCoordinateWrapping::REPEAT: return GL_REPEAT;
                case TextureCoordinateWrapping::NEAREST: return GL_NEAREST;
                case TextureCoordinateWrapping::LINEAR: return GL_LINEAR;
                case TextureCoordinateWrapping::MIRRORED_REPEAT: return GL_MIRRORED_REPEAT;
                case TextureCoordinateWrapping::CLAMP_TO_EDGE: return GL_CLAMP_TO_EDGE;
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
    Texture::Texture(const TextureConfig & config, const void * data, bool initHandle) {
        _impl = std::make_shared<TextureImpl>(config, data, initHandle);
    }

    Texture::~Texture() {}

    void Texture::setCoordinateWrapping(TextureCoordinateWrapping wrap) { _impl->setCoordinateWrapping(wrap); }
    void Texture::setMinMagFilter(TextureMinificationFilter min, TextureMagnificationFilter mag) { _impl->setMinMagFilter(min, mag); }
    void Texture::setTextureCompare(TextureCompareMode mode, TextureCompareFunc func) { _impl->setTextureCompare(mode, func); }

    TextureType Texture::type() const { return _impl->type(); }
    TextureComponentFormat Texture::format() const { return _impl->format(); }
    TextureHandle Texture::handle() const { return _impl->handle(); }

    uint32_t Texture::width() const { return _impl->width(); }
    uint32_t Texture::height() const { return _impl->height(); }
    uint32_t Texture::depth() const { return _impl->depth(); }

    void Texture::bind(int activeTexture) const { _impl->bind(activeTexture); }
    void Texture::unbind() const { _impl->unbind(); }
    bool Texture::valid() const { return _impl != nullptr; }

    const void * Texture::underlying() const { return _impl->underlying(); }

    size_t Texture::hashCode() const {
        return std::hash<void *>{}((void *)_impl.get());
    }

    bool Texture::operator==(const Texture & other) const {
        return _impl == other._impl;
    }

    // Creates a new texture and copies this texture into it
    Texture Texture::copy(uint32_t newWidth, uint32_t newHeight) {
        throw std::runtime_error("Must implement");
    }

    const TextureConfig & Texture::getConfig() const {
        return _impl->getConfig();
    }

    void Texture::_setHandle(const TextureHandle handle) {
        _impl->setHandle(handle);
    }
}