

#ifndef STRATUSGFX_COMMON_H
#define STRATUSGFX_COMMON_H

#include "GL/gl3w.h"
#include "SDL.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include <exception>
#include <stdexcept>

#define BITMASK64_POW2(offset) (1ull << offset)
#define INSTANCE(type) stratus::##type::Instance()

#endif //STRATUSGFX_COMMON_H
