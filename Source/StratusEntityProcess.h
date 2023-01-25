#pragma once

#include <memory>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include "StratusEntityCommon.h"

namespace stratus {
    // An entity system process signals to the engine that it wants to be called once
    // per frame in order to operate on certain entity data lists
    struct EntityProcess : public std::enable_shared_from_this<EntityProcess> {
        virtual ~EntityProcess() = default;

        // Gives the system a change to do whatever processing it needs
        // Guarantee: only one process will be active at a time and may split
        // the entities it is looping over across as many threads as it wants
        //
        // Requirement: when Process returns no entity data is being touched
        // by any other threads
        virtual void Process(const double deltaSeconds) = 0;

        // Called when an entity is added or removed from the world directly,
        // or when it is attached or detached from a parent entity who is
        // part of the world
        virtual void EntitiesAdded(const std::unordered_set<stratus::Entity2Ptr>&) = 0;
        virtual void EntitiesRemoved(const std::unordered_set<stratus::Entity2Ptr>&) = 0;

        // Called when an entity has a component added
        virtual void EntityComponentsAdded(const std::unordered_map<stratus::Entity2Ptr, std::vector<stratus::Entity2Component *>>&) = 0;
    };
}