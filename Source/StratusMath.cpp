#include "StratusMath.h"

namespace stratus {
    Radians::Radians(const Degrees& d) : _rad(glm::radians(d.value())) {}

    Degrees::Degrees(const Radians& r) : _deg(glm::degrees(r.value())) {}

    glm::mat4 Rotation::asMat4() const {
        glm::mat4 m(1.0f);
        matRotate(m, *this);
        return m;
    }

    glm::mat3 Rotation::asMat3() const {
        return glm::mat3(asMat4());
    }
}