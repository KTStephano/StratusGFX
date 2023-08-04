#pragma once

#include <memory>
#include "glm/glm.hpp"
#include "StratusTexture.h"
#include <vector>
#include "StratusTypes.h"

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
        u32 startX;
        u32 startY;
        u32 endX;
        u32 endY;
    };

    class FrameBufferImpl;
    class FrameBuffer {
        std::shared_ptr<FrameBufferImpl> fbo_;

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
        void Clear(const glm::vec4 & rgba);
        void ClearColorLayer(const glm::vec4& rgba, const usize colorIndex, const i32 layer);
        void ClearDepthStencilLayer(const i32 layer);
        // from = rectangular region in *other* to copy from
        // to = rectangular region in *this* to copy to
        void CopyFrom(const FrameBuffer & other, const BufferBounds & from, const BufferBounds & to, BufferBit bit, BufferFilter filter);
        const std::vector<Texture> & GetColorAttachments() const;
        const Texture * GetDepthStencilAttachment() const;

        void Bind() const;
        void Unbind() const;

        // Useful for layered rendering
        void SetColorTextureLayer(const i32 attachmentNum, const i32 mipLevel, const i32 layer);
        void SetDepthTextureLayer(const i32 layer);

        bool Valid() const;
        void * Underlying() const;
    };
}