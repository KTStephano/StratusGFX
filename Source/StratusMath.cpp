#include "StratusMath.h"

namespace stratus {
    Radians::Radians(const Degrees& d) : _rad(glm::radians(d.value())) {}

    Degrees::Degrees(const Radians& r) : _deg(glm::degrees(r.value())) {}
}