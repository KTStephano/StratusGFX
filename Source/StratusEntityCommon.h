#pragma once

#include <memory>

namespace stratus {
    class Entity;
    struct EntityComponent;
    struct EntityComponentSet;
    struct EntityProcess;

    typedef std::shared_ptr<Entity> EntityPtr;
    typedef std::weak_ptr<Entity> EntityWeakPtr;
    typedef std::shared_ptr<EntityProcess> EntityProcessPtr;

    typedef void * EntityProcessHandle;
    constexpr EntityProcessHandle NullEntityProcessHandle = nullptr;
}