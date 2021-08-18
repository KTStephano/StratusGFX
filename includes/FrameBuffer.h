#pragma once

#include <memory>
#include "glm/glm.hpp"
#include "Texture.h"
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
        int startX;
        int startY;
        int endX;
        int endY;
    };

    class FrameBufferImpl;
    class FrameBuffer {
        std::shared_ptr<FrameBufferImpl> _fbo;

    public:
        FrameBuffer();
        ~FrameBuffer();

        FrameBuffer(const FrameBuffer &) = default;
        FrameBuffer(FrameBuffer &&) = default;
        FrameBuffer & operator=(const FrameBuffer &) = default;
        FrameBuffer & operator=(FrameBuffer &&) = default;

        // Clears the color, depth and stencil buffers using rgba
        void clear(const glm::vec4 & rgba) const;
        // Attaches a set of textures to the buffer - only call this once per buffer to avoid runtime exceptions
        void setAttachments(const std::vector<Texture> &);
        // from = rectangular region in *other* to copy from
        // to = rectangular region in *this* to copy to
        void copyFrom(const FrameBuffer & other, const BufferBounds & from, const BufferBounds & to, BufferBit bit, BufferFilter filter);
        const std::vector<Texture> & getColorAttachments() const;
        const Texture & getDepthStencilAttachment() const;

        void bind() const;
        void unbind() const;

        bool valid() const;

        void * underlying() const;
    };
}