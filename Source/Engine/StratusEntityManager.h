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
        void RegisterEntityProcess_(EntityProcessPtr&);
        void AddEntity_(const EntityPtr&);
        void RemoveEntity_(const EntityPtr&);

    private:
        // Meant to be called by Entity
        void NotifyComponentsAdded_(const EntityPtr&, EntityComponent *);
        void NotifyComponentsEnabledDisabled_(const EntityPtr&);

    private:
        mutable std::shared_mutex m_;
        // All entities currently tracked
        std::unordered_set<EntityPtr> entities_;
        // Entities added within the last frame
        std::unordered_set<EntityPtr> entitiesToAdd_;
        // Entities which are pending removal (removed during Update)
        std::unordered_set<EntityPtr> entitiesToRemove_;
        // Processes removed within last frame
        std::unordered_set<EntityProcessHandle> processesToRemove_;
        // Processes added within last frame
        std::vector<EntityProcessPtr> processesToAdd_;
        // Systems which operate on entities
        std::vector<EntityProcessPtr> processes_;
        // Convert handle to process ptr
        std::unordered_map<EntityProcessHandle, EntityProcessPtr> handlesToPtrs_;
        // Component change lists
        std::unordered_map<EntityPtr, std::vector<EntityComponent *>> addedComponents_;
        std::unordered_set<EntityPtr> componentsEnabledDisabled_;
    };

    template<typename E, typename ... Types>
    EntityProcessHandle EntityManager::RegisterEntityProcess(const Types&... args) {
        static_assert(std::is_base_of<EntityProcess, E>::value);
        EntityProcess * p = dynamic_cast<EntityProcess *>(new E(args...));
        EntityProcessPtr ptr(p);
        RegisterEntityProcess_(ptr);
        return (EntityProcessHandle)p;
    }
}