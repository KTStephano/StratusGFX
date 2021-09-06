

#ifndef STRATUSGFX_COMMON_H
#define STRATUSGFX_COMMON_H

#include "GL/gl3w.h"
#include "SDL.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include <exception>
#include <stdexcept>

#define BITMASK64_POW2(offset) (1ull << offset)

namespace stratus {
// typedef int TextureHandle;
// typedef int ShadowMapHandle;
typedef void * RenderDataHandle;

/**
 * This class includes all of the most common functions
 * that need to be references from everywhere. This includes
 * things like print/error print or logging messages to
 * the log file.
 */
struct Common {

};
}

#endif //STRATUSGFX_COMMON_H
