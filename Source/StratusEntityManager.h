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
        friend struct EntityComponentSet;
        friend class Entity;

        ~EntityManager() = default;

        // Add/Remove entities
        void AddEntity(const EntityPtr&);
        void RemoveEntity(const EntityPtr&);

        // Registers or Unregisters an EntityProcess type
        template<typename E, typename ... Types>
        EntityProcessHandle RegisterEntityProcess(const Types&... args);
        void UnregisterEntityProcess(EntityProcessHandle);

        // SystemModule inteface
    private:
        bool Initialize() override;
        SystemStatus Update(const double) override;
        void Shutdown() override;

    private:
        void _RegisterEntityProcess(EntityProcessPtr&);
        void _AddEntity(const EntityPtr&);
        void _RemoveEntity(const EntityPtr&);

    private:
        // Meant to be called by Entity
        void _NotifyComponentsAdded(const EntityPtr&, EntityComponent *);
        void _NotifyComponentsEnabledDisabled(const EntityPtr&);

    private:
        mutable std::shared_mutex _m;
        // All entities currently tracked
        std::unordered_set<EntityPtr> _entities;
        // Entities added within the last frame
        std::unordered_set<EntityPtr> _entitiesToAdd;
        // Entities which are pending removal (removed during Update)
        std::unordered_set<EntityPtr> _entitiesToRemove;
        // Processes removed within last frame
        std::unordered_set<EntityProcessHandle> _processesToRemove;
        // Processes added within last frame
        std::vector<EntityProcessPtr> _processesToAdd;
        // Systems which operate on entities
        std::vector<EntityProcessPtr> _processes;
        // Convert handle to process ptr
        std::unordered_map<EntityProcessHandle, EntityProcessPtr> _handlesToPtrs;
        // Component change lists
        std::unordered_map<EntityPtr, std::vector<EntityComponent *>> _addedComponents;
        std::unordered_set<EntityPtr> _componentsEnabledDisabled;
    };

    template<typename E, typename ... Types>
    EntityProcessHandle EntityManager::RegisterEntityProcess(const Types&... args) {
        static_assert(std::is_base_of<EntityProcess, E>::value);
        EntityProcess * p = dynamic_cast<EntityProcess *>(new E(args...));
        EntityProcessPtr ptr(p);
        _RegisterEntityProcess(ptr);
        return (EntityProcessHandle)p;
    }
}