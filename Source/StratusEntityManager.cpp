#include "StratusEntityManager.h"

/*
    class EntityManager : public SystemModule {
        friend class Engine;
        friend class Entity2;

        EntityManager() {}

    public:
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
        const char * Name() const {
            return "EntityManager";
        }

    private:
        bool Initialize();
        SystemStatus Update(const double);
        void Shutdown();

    private:
        void _RegisterEntityProcess(EntityProcessPtr&);

    private:
        // Meant to be called by Entity
        void NotifyComponentsAdded(const Entity2Ptr&, const std::string&);
        void NotifyComponentsRemoved(const Entity2Ptr&, const std::string&);

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
*/

namespace stratus {
    EntityManager * EntityManager::_instance = nullptr;

    EntityManager::EntityManager() {}

    void EntityManager::AddEntity(Entity2Ptr&) {

    }

    void EntityManager::RemoveEntity(const Entity2Ptr&) {

    }

    bool EntityManager::Initialize() {
        return true;
    }
    
    SystemStatus EntityManager::Update(const double) {
        return SystemStatus::SYSTEM_CONTINUE;
    }
    
    void EntityManager::Shutdown() {

    }
    
    void EntityManager::_RegisterEntityProcess(EntityProcessPtr&) {

    }
    
    void EntityManager::_NotifyComponentsAdded(const Entity2Ptr&, const std::string&) {

    }
    
    void EntityManager::_NotifyComponentsRemoved(const Entity2Ptr&, const std::string&) {

    }
}