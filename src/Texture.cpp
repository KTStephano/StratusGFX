#include "Texture.h"
#include <GL/gl3w.h>
#include <exception>

namespace stratus {
    class TextureImpl {
        GLuint _texture;
        TextureType _type;
        TextureComponentFormat _format;
        uint32_t _width;
        uint32_t _height;

    public:
        TextureImpl(const TextureConfig & config, const void * data) {
            glGenTextures(1, &_texture);

            _type = config.type;
            _format = config.format;
            _width = config.width;
            _height = config.height;

            _bind();
            if (config.type == TextureType::TEXTURE_2D) {
                glTexImage2D(GL_TEXTURE_2D,
                    0,
                    _convertInternalFormat(config.format, config.storage, config.dataType),
                    config.width, config.height,
                    0,
                    _convertFormat(config.format),
                    _convertType(config.dataType, config.storage),
                    data);
                if (config.generateMipMaps) (GL_TEXTURE_2D);
            }
            else {
                for (int face = 0; face < 6; ++face) {
                    glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + face, 
                        0, 
                        _convertInternalFormat(config.format, config.storage, config.dataType),
                        config.width, config.height, 
                        0, 
                        _convertFormat(config.format),
                        _convertType(config.dataType, config.storage), 
                        data);
                }
            }
            _unbind();
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
            glTexParameteri(_convertTexture(_type), GL_TEXTURE_WRAP_S, _convertTextureCoordinateWrapping(wrap));
            glTexParameteri(_convertTexture(_type), GL_TEXTURE_WRAP_T, _convertTextureCoordinateWrapping(wrap));
            // Support third dimension for cube maps
            if (_type == TextureType::TEXTURE_3D) glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, _convertTextureCoordinateWrapping(wrap));
        }

        void setMinMagFilter(TextureMinificationFilter min, TextureMagnificationFilter mag) {
            glTexParameteri(_convertTexture(_type), GL_TEXTURE_MIN_FILTER, _convertTextureMinFilter(min));
            glTexParameteri(_convertTexture(_type), GL_TEXTURE_MAG_FILTER, _convertTextureMagFilter(mag));
        }

        TextureType type() const { return _type; }

        TextureComponentFormat format() const { return _format; }

        uint32_t width() const { return _width; }

        uint32_t height() const { return _height; }

    private:
        void _bind() const {
            glBindTexture(_convertTexture(_type), _texture);
        }

        void _unbind() const {
            glBindTexture(_convertTexture(_type), 0);
        }

        static GLenum _convertTexture(TextureType type) {
            if (type == TextureType::TEXTURE_2D) {
                return GL_TEXTURE_2D;
            }
            else {
                return GL_TEXTURE_CUBE_MAP;
            }
        }

        static GLenum _convertFormat(TextureComponentFormat format) {
            switch (format) {
                case TextureComponentFormat::RED: return GL_RED;
                case TextureComponentFormat::RGB: return GL_RGB;
                case TextureComponentFormat::SRGB: return GL_SRGB;
                case TextureComponentFormat::RGBA: return GL_RGBA;
                case TextureComponentFormat::SRGB_ALPHA: return GL_SRGB_ALPHA;
                case TextureComponentFormat::DEPTH: return GL_DEPTH_COMPONENT;
                case TextureComponentFormat::DEPTH_STENCIL: return GL_DEPTH_STENCIL;
                default: throw std::runtime_error("Unknown format");
            }
        }

        static GLint _convertInternalFormat(TextureComponentFormat format, TextureComponentSize size, TextureComponentType type) {
            // If the bits are default we just mirror the format for the internal format option
            if (format == TextureComponentFormat::DEPTH || format == TextureComponentFormat::DEPTH_STENCIL || size == TextureComponentSize::BITS_DEFAULT) {
                return (GLint)_convertFormat(format);
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
    };

    Texture::Texture() {}
    Texture::Texture(const TextureConfig & config, const void * data) {
        _impl = std::make_shared<TextureImpl>(config, data);
    }

    Texture::~Texture() {}

    void Texture::setCoordinateWrapping(TextureCoordinateWrapping wrap) { _impl->setCoordinateWrapping(wrap); }
    void Texture::setMinMagFilter(TextureMinificationFilter min, TextureMagnificationFilter mag) { _impl->setMinMagFilter(min, mag); }

    TextureType Texture::type() const { return _impl->type(); }
    TextureComponentFormat Texture::format() const { return _impl->format(); }

    uint32_t Texture::width() const { return _impl->width(); }
    uint32_t Texture::height() const { return _impl->height(); }
}