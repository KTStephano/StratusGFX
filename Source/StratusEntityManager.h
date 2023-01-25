#pragma once

#include "StratusHandle.h"
#include "StratusSystemModule.h"
#include <memory>
#include <unordered_set>
#include <unordered_map>
#include <shared_mutex>
#include <cstdint>
#include <string>
#include <vector>
#include "StratusEntityCommon.h"
#include "StratusEntityProcess.h"

namespace stratus {
    SYSTEM_MODULE_CLASS(EntityManager)
        friend struct Entity2ComponentSet;
        friend class Entity2;

        ~EntityManager() = default;

        // Add/Remove entities
        void AddEntity(const Entity2Ptr&);
        void RemoveEntity(const Entity2Ptr&);

        // Registers an EntityProcess type
        template<typename E, typename ... Types>
        void RegisterEntityProcess(Types ... args);

        // SystemModule inteface
    private:
        bool Initialize() override;
        SystemStatus Update(const double) override;
        void Shutdown() override;

    private:
        void _RegisterEntityProcess(EntityProcessPtr&);

    private:
        // Meant to be called by Entity
        void _NotifyComponentsAdded(const Entity2Ptr&, Entity2Component *);

    private:
        mutable std::shared_mutex _m;
        // All entities currently tracked
        std::unordered_set<Entity2Ptr> _entities;
        // Entities added within the last frame
        std::unordered_set<Entity2Ptr> _entitiesToAdd;
        // Entities which are pending removal (removed during Update)
        std::unordered_set<Entity2Ptr> _entitiesToRemove;
        // Processes added within last frame
        std::vector<EntityProcessPtr> _processesToAdd;
        // Systems which operate on entities
        std::vector<EntityProcessPtr> _processes;
        // Component change lists
        std::unordered_map<Entity2Ptr, std::vector<Entity2Component *>> _addedComponents;
    };

    template<typename E, typename ... Types>
    void EntityManager::RegisterEntityProcess(Types ... args) {
        static_assert(std::is_base_of<EntityProcess, E>::value);
        EntityProcess * p = dynamic_cast<EntityProcess *>(new E(std::forward<Types>(args)...));
        EntityProcessPtr ptr(p);
        _RegisterEntityProcess(ptr);
    }
}