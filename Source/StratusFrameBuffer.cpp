#include "StratusFrameBuffer.h"
#include "GL/gl3w.h"
#include <iostream>

namespace stratus {
   class FrameBufferImpl {
       GLuint _fbo;
       std::vector<Texture> _colorAttachments;
       Texture _depthStencilAttachment;
       mutable GLenum _currentBindingPoint = 0;
       bool _valid = false;
    
    public:
        FrameBufferImpl() {
            glGenFramebuffers(1, &_fbo);
        }

        ~FrameBufferImpl() {
            glDeleteFramebuffers(1, &_fbo);
        }

        // No copying
        FrameBufferImpl(const FrameBufferImpl &) = delete;
        FrameBufferImpl(FrameBufferImpl &&) = delete;
        FrameBufferImpl & operator=(const FrameBufferImpl &) = delete;
        FrameBufferImpl & operator=(FrameBufferImpl &&) = delete;

        void clear(const glm::vec4 & rgba) const {
            if (_currentBindingPoint != 0) std::cerr << "Warning: clear() called after bind()" << std::endl;
            bind();
            glClearColor(rgba.r, rgba.g, rgba.b, rgba.a);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
            unbind();
        }

        void bind() const {
            _bind(GL_FRAMEBUFFER);
        }

        void unbind() const {
            if (_currentBindingPoint == -1) return;
            glBindFramebuffer(_currentBindingPoint, 0);
            _currentBindingPoint = 0;
        }

        void setAttachments(const std::vector<Texture> & attachments) {
            if (_colorAttachments.size() > 0 || _depthStencilAttachment.valid()) throw std::runtime_error("setAttachments called twice");
            _valid = true;

            bind();

            // We can only have 1 max for each
            int numDepthStencilAttachments = 0;

            // In the case of multiple color attachments we need to let OpenGL know
            std::vector<uint32_t> drawBuffers;

            for (Texture tex : attachments) {
                tex.bind();
                GLuint underlying = *(GLuint *)tex.underlying();
                if (tex.format() == TextureComponentFormat::DEPTH) {
                    if (numDepthStencilAttachments > 0) throw std::runtime_error("More than one depth attachment present");
                    /*
                    glFramebufferTexture2D(GL_FRAMEBUFFER,
                        GL_DEPTH_ATTACHMENT,
                        GL_TEXTURE_2D,
                        underlying,
                        0);
                    */
                    glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, underlying, 0);
                    ++numDepthStencilAttachments;
                    _depthStencilAttachment = tex;
                }
                else if (tex.format() == TextureComponentFormat::DEPTH_STENCIL) {
                    if (numDepthStencilAttachments > 0) throw std::runtime_error("More than one depth_stencil attachment present");
                    /*
                    glFramebufferTexture2D(GL_FRAMEBUFFER,
                        GL_DEPTH_STENCIL_ATTACHMENT,
                        GL_TEXTURE_2D,
                        underlying,
                        0);
                    */
                    glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, underlying, 0);
                    ++numDepthStencilAttachments;
                    _depthStencilAttachment = tex;
                }
                else {
                    GLenum color = GL_COLOR_ATTACHMENT0 + drawBuffers.size();
                    drawBuffers.push_back(color);
                    /*
                    glFramebufferTexture2D(GL_FRAMEBUFFER, 
                        color, 
                        GL_TEXTURE_2D, 
                        underlying, 
                        0);
                    */
                    glFramebufferTexture(GL_FRAMEBUFFER, color, underlying, 0);
                    _colorAttachments.push_back(tex);
                }
                tex.unbind();
            }

            if (drawBuffers.size() == 0) {
                // Tell OpenGL we won't be using a color buffer
                glDrawBuffer(GL_NONE);
                glReadBuffer(GL_NONE);
            }
            else {
                glDrawBuffers(drawBuffers.size(), &drawBuffers[0]);
            }

            // Validity check
            if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
                std::cerr << "[error] Generating frame buffer with attachments failed" << std::endl;
                _valid = false;
            }

            unbind();
        }

        bool valid() const {
            return _valid;
        }

        void copyFrom(const FrameBufferImpl & other, const BufferBounds & from, const BufferBounds & to, BufferBit bit, BufferFilter filter) {
            other._bind(GL_READ_FRAMEBUFFER); // read from
            this->_bind(GL_DRAW_FRAMEBUFFER); // write to
            // Blit to default framebuffer - not that the framebuffer you are writing to has to match the internal format
            // of the framebuffer you are reading to!
            GLbitfield mask = 0;
            if (bit & COLOR_BIT) mask |= GL_COLOR_BUFFER_BIT;
            if (bit & DEPTH_BIT) mask |= GL_DEPTH_BUFFER_BIT;
            if (bit & STENCIL_BIT) mask |= GL_STENCIL_BUFFER_BIT;
            GLenum blitFilter = (filter == BufferFilter::NEAREST) ? GL_NEAREST : GL_LINEAR;
            glBlitFramebuffer(from.startX, from.startY, from.endX, from.endY, to.startX, to.startY, to.endX, to.endY, mask, blitFilter);
            other.unbind();
            this->unbind();
        }

        const std::vector<Texture> & getColorAttachments() const {
            return _colorAttachments;
        }

        const Texture * getDepthStencilAttachment() const {
            return &_depthStencilAttachment;
        }

        void * underlying() const {
            return (void *)&_fbo;
        }

    private:
        void _bind(GLenum bindingPoint) const {
            glBindFramebuffer(bindingPoint, _fbo);
            _currentBindingPoint = bindingPoint;
        }
    };

    FrameBuffer::FrameBuffer() {}
    FrameBuffer::FrameBuffer(const std::vector<Texture> & attachments) {
        _fbo = std::make_shared<FrameBufferImpl>();
        _fbo->setAttachments(attachments);
    }
    FrameBuffer::~FrameBuffer() {}

    // Clears the color, depth and stencil buffers using rgba
    void FrameBuffer::clear(const glm::vec4 & rgba) const {
        _fbo->clear(rgba);
    }

    // from = rectangular region in *other* to copy from
    // to = rectangular region in *this* to copy to
    void FrameBuffer::copyFrom(const FrameBuffer & other, const BufferBounds & from, const BufferBounds & to, BufferBit bit, BufferFilter filter) {
        _fbo->copyFrom(*other._fbo, from, to, bit, filter);
    }
    
    const std::vector<Texture> & FrameBuffer::getColorAttachments() const { return _fbo->getColorAttachments(); }
    const Texture * FrameBuffer::getDepthStencilAttachment() const        { return _fbo->getDepthStencilAttachment(); }

    void FrameBuffer::bind() const         { _fbo->bind(); }
    void FrameBuffer::unbind() const       { _fbo->unbind(); }
    bool FrameBuffer::valid() const        { return _fbo->valid(); }
    void * FrameBuffer::underlying() const { return _fbo->underlying(); }
}