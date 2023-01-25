#pragma once

#include <memory>

namespace stratus {
    class Entity2;
    struct Entity2Component;
    struct Entity2ComponentSet;
    struct EntityProcess;

    typedef std::shared_ptr<Entity2> Entity2Ptr;
    typedef std::weak_ptr<Entity2> Entity2WeakPtr;
    typedef std::shared_ptr<EntityProcess> EntityProcessPtr;
}