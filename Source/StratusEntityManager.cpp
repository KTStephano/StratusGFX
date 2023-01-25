#include "StratusEntityManager.h"
#include "StratusEntity2.h"
#include "StratusApplicationThread.h"

namespace stratus {
    EntityManager::EntityManager() {}

    void EntityManager::AddEntity(Entity2Ptr& e) {
        std::unique_lock<std::shared_mutex> ul(_m);
        _entitiesToAdd.insert(std::move(e));
    }

    void EntityManager::RemoveEntity(const Entity2Ptr& e) {
        std::unique_lock<std::shared_mutex> ul(_m);
        _entitiesToRemove.insert(e);
    }

    bool EntityManager::Initialize() {
        return true;
    }
    
    SystemStatus EntityManager::Update(const double deltaSeconds) {
        CHECK_IS_APPLICATION_THREAD();

        // Notify processes of added/removed entities and allow them to
        // perform their process routine
        auto entitiesToAdd = std::move(_entitiesToAdd);
        auto entitiesToRemove = std::move(_entitiesToRemove);
        for (EntityProcessPtr& ptr : _processes) {
            ptr->EntitiesAdded(entitiesToAdd);
            ptr->EntitiesRemoved(entitiesToRemove);
            ptr->Process(deltaSeconds);
        }

        // Commit added/removed entities
        _entities.insert(entitiesToAdd.begin(), entitiesToAdd.end());
        _entities.erase(entitiesToRemove.begin(), entitiesToRemove.end());

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
        _entities.clear();
        _entitiesToAdd.clear();
        _entitiesToRemove.clear();
        _processes.clear();
        _processesToAdd.clear();
    }
    
    void EntityManager::_RegisterEntityProcess(EntityProcessPtr& ptr) {
        std::unique_lock<std::shared_mutex> ul(_m);
        _processesToAdd.push_back(std::move(ptr));
    }
    
    void EntityManager::_NotifyComponentsAdded(const Entity2Ptr& ptr, Entity2Component * component) {

    }
}