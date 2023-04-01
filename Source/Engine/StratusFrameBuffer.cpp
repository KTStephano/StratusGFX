#include "StratusFrameBuffer.h"
#include "GL/gl3w.h"
#include <iostream>
#include "StratusLog.h"
#include "StratusApplicationThread.h"

namespace stratus {
    class FrameBufferImpl {
        GLuint fbo_;
        std::vector<Texture> colorAttachments_;
        std::vector<GLenum> glColorAttachments_; // For use with glDrawBuffers
        Texture depthStencilAttachment_;
        mutable GLenum currentBindingPoint_ = 0;
        bool valid_ = false;

    public:
        FrameBufferImpl() {
            glGenFramebuffers(1, &fbo_);
        }

        ~FrameBufferImpl() {
            if (ApplicationThread::Instance()->CurrentIsApplicationThread()) {
                glDeleteFramebuffers(1, &fbo_);
            }
            else {
                auto buffer = fbo_;
                ApplicationThread::Instance()->Queue([buffer]() { auto fbo = buffer; glDeleteFramebuffers(1, &fbo); });
            }
        }

        // No copying
        FrameBufferImpl(const FrameBufferImpl &) = delete;
        FrameBufferImpl(FrameBufferImpl &&) = delete;
        FrameBufferImpl & operator=(const FrameBufferImpl &) = delete;
        FrameBufferImpl & operator=(FrameBufferImpl &&) = delete;

        void Clear(const glm::vec4 & rgba) {
            bool bindAndUnbind = true;
            if (currentBindingPoint_ != 0) bindAndUnbind = false;
            if (bindAndUnbind) Bind();
            glDepthMask(GL_TRUE);
            glStencilMask(GL_TRUE);
            glDrawBuffers(glColorAttachments_.size(), glColorAttachments_.data());
            glClearColor(rgba.r, rgba.g, rgba.b, rgba.a);
            glClearDepthf(1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
            if (bindAndUnbind) Unbind();
        }

        void ClearColorLayer(const glm::vec4& rgba, const size_t colorIndex, const int layer) {
            if (colorAttachments_.size() < colorIndex) {
                throw std::runtime_error("Color index exceeds maximum total bound color buffers");
            }
            
            // TODO: There is a much better way to do this
            // See https://registry.khronos.org/OpenGL-Refpages/gl4/html/glFramebufferTextureLayer.xhtml
            // Followed by a regular glClear of the color attachment
            Texture& color = colorAttachments_[colorIndex];
            color.ClearLayer(0, layer, (const void *)&rgba[0]);
        }

        void ClearDepthStencilLayer(const int layer) {
            if (depthStencilAttachment_ == Texture()) {
                throw std::runtime_error("Attempt to clear null depth/stencil attachment");
            }

            float val = 1.0f;
            // TODO: There is a much better way to do this
            // See https://registry.khronos.org/OpenGL-Refpages/gl4/html/glFramebufferTextureLayer.xhtml
            // Followed by a regular glClear of the depth stencil attachment
            //std::vector<float> data(_depthStencilAttachment.width() * _depthStencilAttachment.height(), val);
            depthStencilAttachment_.ClearLayer(0, layer, (const void *)&val);
        }

        void Bind() const {
            _bind(GL_FRAMEBUFFER);
        }

        void Unbind() const {
            if (currentBindingPoint_ == -1) return;
            glBindFramebuffer(currentBindingPoint_, 0);
            currentBindingPoint_ = 0;
        }

        void SetColorTextureLayer(const int attachmentNum, const int mipLevel, const int layer) {
            if (colorAttachments_.size() < attachmentNum) {
                throw std::runtime_error("Attachment number exceeds amount of attached color textures");
            }

            glNamedFramebufferTextureLayer(
                fbo_, 
                GL_COLOR_ATTACHMENT0 + attachmentNum,
                *(GLuint *)GetColorAttachments()[attachmentNum].Underlying(),
                mipLevel, layer
            );
        }

        void SetDepthTextureLayer(const int layer) {
            if (depthStencilAttachment_ == Texture()) {
                throw std::runtime_error("Attempt to use null depth/stencil attachment");
            }

            glNamedFramebufferTextureLayer(
                fbo_, 
                GL_DEPTH_ATTACHMENT,
                *(GLuint *)GetDepthStencilAttachment()->Underlying(),
                0, layer
            );
        }

        void setAttachments(const std::vector<Texture> & attachments) {
            if (colorAttachments_.size() > 0 || depthStencilAttachment_.Valid()) throw std::runtime_error("setAttachments called twice");
            valid_ = true;

            Bind();

            // We can only have 1 max for each
            int numDepthStencilAttachments = 0;

            // In the case of multiple color attachments we need to let OpenGL know
            std::vector<uint32_t> drawBuffers;

            for (Texture tex : attachments) {
                tex.Bind();
                GLuint Underlying = *(GLuint *)tex.Underlying();
                if (tex.Format() == TextureComponentFormat::DEPTH) {
                    if (numDepthStencilAttachments > 0) throw std::runtime_error("More than one depth attachment present");
                    /*
                    glFramebufferTexture2D(GL_FRAMEBUFFER,
                        GL_DEPTH_ATTACHMENT,
                        GL_TEXTURE_2D,
                        underlying,
                        0);
                    */
                    glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, Underlying, 0);
                    ++numDepthStencilAttachments;
                    depthStencilAttachment_ = tex;
                }
                else if (tex.Format() == TextureComponentFormat::DEPTH_STENCIL) {
                    if (numDepthStencilAttachments > 0) throw std::runtime_error("More than one depth_stencil attachment present");
                    /*
                    glFramebufferTexture2D(GL_FRAMEBUFFER,
                        GL_DEPTH_STENCIL_ATTACHMENT,
                        GL_TEXTURE_2D,
                        underlying,
                        0);
                    */
                    glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, Underlying, 0);
                    ++numDepthStencilAttachments;
                    depthStencilAttachment_ = tex;
                }
                else {
                    GLenum color = GL_COLOR_ATTACHMENT0 + drawBuffers.size();
                    glColorAttachments_.push_back(color);
                    drawBuffers.push_back(color);
                    /*
                    glFramebufferTexture2D(GL_FRAMEBUFFER, 
                        color, 
                        GL_TEXTURE_2D, 
                        underlying, 
                        0);
                    */
                    glFramebufferTexture(GL_FRAMEBUFFER, color, Underlying, 0);
                    colorAttachments_.push_back(tex);
                }
                tex.Unbind();
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
                valid_ = false;
            }

            Unbind();
        }

        bool Valid() const {
            return valid_;
        }

        void CopyFrom(const FrameBufferImpl & other, const BufferBounds & from, const BufferBounds & to, BufferBit bit, BufferFilter filter) {
            // Blit to default framebuffer - not that the framebuffer you are writing to has to match the internal format
            // of the framebuffer you are reading to!
            GLbitfield mask = 0;
            if (bit & COLOR_BIT) mask |= GL_COLOR_BUFFER_BIT;
            if (bit & DEPTH_BIT) mask |= GL_DEPTH_BUFFER_BIT;
            if (bit & STENCIL_BIT) mask |= GL_STENCIL_BUFFER_BIT;
            GLenum blitFilter = (filter == BufferFilter::NEAREST) ? GL_NEAREST : GL_LINEAR;
            glBlitNamedFramebuffer(other.fbo_, fbo_, from.startX, from.startY, from.endX, from.endY, to.startX, to.startY, to.endX, to.endY, mask, blitFilter);
        }

        const std::vector<Texture> & GetColorAttachments() const {
            return colorAttachments_;
        }

        const Texture * GetDepthStencilAttachment() const {
            return &depthStencilAttachment_;
        }

        void * Underlying() const {
            return (void *)&fbo_;
        }

    private:
        void _bind(GLenum bindingPoint) const {
            glBindFramebuffer(bindingPoint, fbo_);
            currentBindingPoint_ = bindingPoint;
        }
    };

    FrameBuffer::FrameBuffer() {}
    FrameBuffer::FrameBuffer(const std::vector<Texture> & attachments) {
        fbo_ = std::make_shared<FrameBufferImpl>();
        fbo_->setAttachments(attachments);
    }
    FrameBuffer::~FrameBuffer() {}

    // Clears the color, depth and stencil buffers using rgba
    void FrameBuffer::Clear(const glm::vec4 & rgba) {
        fbo_->Clear(rgba);
    }

    void FrameBuffer::ClearColorLayer(const glm::vec4& rgba, const size_t colorIndex, const int layer) {
        fbo_->ClearColorLayer(rgba, colorIndex, layer);
    }

    void FrameBuffer::ClearDepthStencilLayer(const int layer) {
        fbo_->ClearDepthStencilLayer(layer);
    }

    // from = rectangular region in *other* to copy from
    // to = rectangular region in *this* to copy to
    void FrameBuffer::CopyFrom(const FrameBuffer & other, const BufferBounds & from, const BufferBounds & to, BufferBit bit, BufferFilter filter) {
        fbo_->CopyFrom(*other.fbo_, from, to, bit, filter);
    }

    const std::vector<Texture> & FrameBuffer::GetColorAttachments() const { return fbo_->GetColorAttachments(); }
    const Texture * FrameBuffer::GetDepthStencilAttachment() const        { return fbo_->GetDepthStencilAttachment(); }

    void FrameBuffer::Bind() const         { fbo_->Bind(); }
    void FrameBuffer::Unbind() const       { fbo_->Unbind(); }
    bool FrameBuffer::Valid() const        { return fbo_ != nullptr && fbo_->Valid(); }
    void * FrameBuffer::Underlying() const { return fbo_->Underlying(); }

    void FrameBuffer::SetColorTextureLayer(const int attachmentNum, const int mipLevel, const int layer) { 
        fbo_->SetColorTextureLayer(attachmentNum, mipLevel, layer); 
    }

    void FrameBuffer::SetDepthTextureLayer(const int layer) { 
        fbo_->SetDepthTextureLayer(layer);
    }
}