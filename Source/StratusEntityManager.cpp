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

    void EntityManager::AddEntity(Entity2Ptr& e) {
        _entitiesToAdd.insert(std::move(e));
    }

    void EntityManager::RemoveEntity(const Entity2Ptr& e) {

    }

    bool EntityManager::Initialize() {
        return true;
    }
    
    SystemStatus EntityManager::Update(const double deltaSeconds) {
        // Notify processes of added/removed entities and allow them to
        // perform their process routine
        for (EntityProcessPtr& ptr : _processes) {
            ptr->EntitiesAdded(_entitiesToAdd);
            ptr->EntitiesRemoved(_entitiesToRemove);
            ptr->Process(deltaSeconds);
        }

        // Commit added/removed entities
        _entities.insert(_entitiesToAdd.begin(), _entitiesToAdd.end());
        _entities.erase(_entitiesToRemove.begin(), _entitiesToRemove.end());
        _entitiesToAdd.clear();
        _entitiesToRemove.clear();

        // If any processes have been added, tell them about all available entities
        // and allow them to perform their process routine for the first time
        auto processesToAdd = std::move(_processesToAdd);
        _processesToAdd.clear();
        for (EntityProcessPtr& ptr : processesToAdd) {
            ptr->EntitiesAdded(_entities);
            ptr->Process(deltaSeconds);

            // Commit process to list
            _processes.push_back(std::move(ptr));
        }

        return SystemStatus::SYSTEM_CONTINUE;
    }
    
    void EntityManager::Shutdown() {

    }
    
    void EntityManager::_RegisterEntityProcess(EntityProcessPtr& ptr) {
        std::unique_lock<std::shared_mutex> ul(_m);
        _processesToAdd.push_back(std::move(ptr));
    }
    
    void EntityManager::_NotifyComponentsAdded(const Entity2Ptr&, const std::string&) {

    }
    
    void EntityManager::_NotifyComponentsRemoved(const Entity2Ptr&, const std::string&) {

    }
}