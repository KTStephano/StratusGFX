#include "StratusEntityManager.h"
#include "StratusEntity2.h"
#include "StratusApplicationThread.h"

namespace stratus {
    EntityManager::EntityManager() {}

    void EntityManager::AddEntity(const Entity2Ptr& e) {
        if (e->GetParentNode() != nullptr) {
            throw std::runtime_error("Unsupported operation - must add root node");
        }
        std::unique_lock<std::shared_mutex> ul(_m);
        _AddEntity(e);
    }

    void EntityManager::RemoveEntity(const Entity2Ptr& e) {
        if (e->GetParentNode() != nullptr) {
            throw std::runtime_error("Unsupported operation - tree structure is immutable after adding to manager");
        }
        std::unique_lock<std::shared_mutex> ul(_m);
        _RemoveEntity(e);
    }

    void EntityManager::_AddEntity(const Entity2Ptr& e) {
        _entitiesToAdd.insert(e);
        for (const Entity2Ptr& c : e->GetChildNodes()) {
            _AddEntity(c);
        }
    }

    void EntityManager::_RemoveEntity(const Entity2Ptr& e) {
        _entitiesToRemove.insert(e);    
        for (const Entity2Ptr& c : e->GetChildNodes()) {
            _RemoveEntity(c);
        }
    }

    bool EntityManager::Initialize() {
        return true;
    }
    
    SystemStatus EntityManager::Update(const double deltaSeconds) {
        CHECK_IS_APPLICATION_THREAD();

        // Notify processes of added/removed entities and allow them to
        // perform their process routine
        auto entitiesToAdd = std::move(_entitiesToAdd);
        auto addedComponents = std::move(_addedComponents);
        auto entitiesToRemove = std::move(_entitiesToRemove);
        auto componentsEnabledDisabled = std::move(_componentsEnabledDisabled);
        for (EntityProcessPtr& ptr : _processes) {
            if (entitiesToAdd.size() > 0) ptr->EntitiesAdded(entitiesToAdd);
            if (addedComponents.size() > 0) ptr->EntityComponentsAdded(addedComponents);
            if (entitiesToRemove.size() > 0) ptr->EntitiesRemoved(entitiesToRemove);
            if (componentsEnabledDisabled.size() > 0) ptr->EntityComponentsEnabledDisabled(componentsEnabledDisabled);
            ptr->Process(deltaSeconds);
        }

        // Commit added/removed entities
        _entities.insert(entitiesToAdd.begin(), entitiesToAdd.end());
        _entities.erase(entitiesToRemove.begin(), entitiesToRemove.end());

        // If any processes have been added, tell them about all available entities
        // and allow them to perform their process routine for the first time
        auto processesToAdd = std::move(_processesToAdd);
        for (EntityProcessPtr& ptr : processesToAdd) {
            if (_entities.size() > 0) ptr->EntitiesAdded(_entities);
            ptr->Process(deltaSeconds);

            // Commit process to list
            _processes.push_back(ptr);
            _handlesToPtrs.insert(std::make_pair((EntityProcessHandle)ptr.get(), ptr));
        }

        // If any processes have been removed then remove them now
        auto processesToRemove = std::move(_processesToRemove);
        for (EntityProcessHandle handle : processesToRemove) {
            auto handleIt = _handlesToPtrs.find(handle);
            if (handleIt == _handlesToPtrs.end()) continue;
            auto remove = handleIt->second;
            for (auto it = _processes.begin(); it != _processes.end(); ++it) {
                EntityProcessPtr process = *it;
                if (process == remove) {
                    _processes.erase(it);
                    break;
                }
            }
            _handlesToPtrs.erase(handle);
        }

        return SystemStatus::SYSTEM_CONTINUE;
    }
    
    void EntityManager::Shutdown() {
        _entities.clear();
        _entitiesToAdd.clear();
        _entitiesToRemove.clear();
        _processes.clear();
        _processesToAdd.clear();
        _addedComponents.clear();
    }
    
    void EntityManager::_RegisterEntityProcess(EntityProcessPtr& ptr) {
        std::unique_lock<std::shared_mutex> ul(_m);
        _processesToAdd.push_back(std::move(ptr));
    }

    void EntityManager::UnregisterEntityProcess(EntityProcessHandle handle) {
        std::unique_lock<std::shared_mutex> ul(_m);
        _processesToRemove.insert(handle);
    }
    
    void EntityManager::_NotifyComponentsAdded(const Entity2Ptr& ptr, Entity2Component * component) {
        std::unique_lock<std::shared_mutex> ul(_m);
        auto it = _addedComponents.find(ptr);
        if (it == _addedComponents.end()) {
            _addedComponents.insert(std::make_pair(ptr, std::vector<Entity2Component *>{component}));
        }
        else {
            it->second.push_back(component);
        }
    }

    void EntityManager::_NotifyComponentsEnabledDisabled(const Entity2Ptr& ptr) {
        std::unique_lock<std::shared_mutex> ul(_m);
        _componentsEnabledDisabled.insert(ptr);
    }
}