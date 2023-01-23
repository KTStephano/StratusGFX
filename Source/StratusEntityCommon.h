#pragma once

#include <memory>

namespace stratus {
    class Entity2;
    struct EntityProcess;

    typedef std::shared_ptr<Entity2> Entity2Ptr;
    typedef std::shared_ptr<EntityProcess> EntityProcessPtr;
}