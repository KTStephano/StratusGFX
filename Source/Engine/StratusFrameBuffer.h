#pragma once

#include <memory>
#include "glm/glm.hpp"
#include "StratusTexture.h"
#include <vector>

namespace stratus {
    enum BufferBit {
        COLOR_BIT = 1,
        DEPTH_BIT = 2,
        STENCIL_BIT = 4
    };

    enum class BufferFilter {
        NEAREST,
        LINEAR
    };

    struct BufferBounds {
        uint32_t startX;
        uint32_t startY;
        uint32_t endX;
        uint32_t endY;
    };

    class FrameBufferImpl;
    class FrameBuffer {
        std::shared_ptr<FrameBufferImpl> _fbo;

    public:
        FrameBuffer();
        // This constructor sets the attachments
        FrameBuffer(const std::vector<Texture> &);
        ~FrameBuffer();

        FrameBuffer(const FrameBuffer &) = default;
        FrameBuffer(FrameBuffer &&) = default;
        FrameBuffer & operator=(const FrameBuffer &) = default;
        FrameBuffer & operator=(FrameBuffer &&) = default;

        // Clears the color, depth and stencil buffers using rgba
        void clear(const glm::vec4 & rgba);
        void ClearColorLayer(const glm::vec4& rgba, const size_t colorIndex, const int layer);
        void ClearDepthStencilLayer(const int layer);
        // from = rectangular region in *other* to copy from
        // to = rectangular region in *this* to copy to
        void copyFrom(const FrameBuffer & other, const BufferBounds & from, const BufferBounds & to, BufferBit bit, BufferFilter filter);
        const std::vector<Texture> & getColorAttachments() const;
        const Texture * getDepthStencilAttachment() const;

        void Bind() const;
        void Unbind() const;

        // Useful for layered rendering
        void SetColorTextureLayer(const int attachmentNum, const int mipLevel, const int layer);
        void SetDepthTextureLayer(const int layer);

        bool valid() const;
        void * underlying() const;
    };
}