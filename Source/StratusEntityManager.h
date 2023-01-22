#pragma once

#include "StratusHandle.h"
#include "StratusSystemModule.h"
#include <memory>
#include <unordered_set>
#include <shared_mutex>
#include <cstdint>
#include <string>
#include <vector>
#include "StratusEntityCommon.h"
#include "StratusEntityProcess.h"

namespace stratus {
    SYSTEM_MODULE_CLASS(EntityManager)
        friend class Entity2;

        ~EntityManager() = default;

        static EntityManager * Instance() { return _instance; }

        // Add/Remove entities
        // Add will cause the pointer to be set to null as EntityManager
        // takes full ownership
        void AddEntity(Entity2Ptr&);
        void RemoveEntity(const Entity2Ptr&);

        // Registers an EntityProcess type
        template<typename E, typename ... Types>
        void RegisterEntityProcess(Types ... args);

        // SystemModule inteface
        const char * Name() const override {
            return "EntityManager";
        }

    private:
        bool Initialize() override;
        SystemStatus Update(const double) override;
        void Shutdown() override;

    private:
        void _RegisterEntityProcess(EntityProcessPtr&);

    private:
        // Meant to be called by Entity
        void _NotifyComponentsAdded(const Entity2Ptr&, const std::string&);
        void _NotifyComponentsRemoved(const Entity2Ptr&, const std::string&);

    private:
        static EntityManager * _instance;

    private:
        mutable std::shared_mutex _m;
        // All entities currently tracked
        std::unordered_set<Entity2Ptr> _entities;
        // Entities added within the last frame
        std::unordered_set<Entity2Ptr> _entitiesToAdd;
        // Entities which are pending removal (removed during Update)
        std::unordered_set<Entity2Ptr> _entitiesToRemove;
        // Systems which operate on entities
        std::vector<EntityProcessPtr> _processes;
    };

    template<typename E, typename ... Types>
    void EntityManager::RegisterEntityProcess(Types ... args) {
        static_assert(std::is_base_of<EntityProcess, E>::value);
        EntityProcessPtr ptr(new E(std::forward<Types>(args)...));
        _RegisterEntityProcess(ptr);
    }
}