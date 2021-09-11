#pragma once

#include "StratusCommon.h"
#include <cstddef>

namespace stratus {
    struct RendererParams {
        uint32_t viewportWidth;
        uint32_t viewportHeight;
    };

    // Public interface of the renderer - manages frame to frame state and manages
    // the backend
    class RendererFrontEnd {
        friend class Engine;
        RendererFrontEnd(const RendererParams&);

    public:

    }
}