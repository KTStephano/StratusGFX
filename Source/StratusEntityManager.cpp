#include "StratusEntityManager.h"
#include "StratusEntity.h"
#include "StratusApplicationThread.h"
#include "StratusTransformComponent.h"
#include <algorithm>

namespace stratus {
    EntityManager::EntityManager() {}

    void EntityManager::AddEntity(const EntityPtr& e) {
        if (e->GetParentNode() != nullptr) {
            throw std::runtime_error("Unsupported operation - must add root node");
        }
        std::unique_lock<std::shared_mutex> ul(_m);
        _AddEntity(e);
    }

    void EntityManager::RemoveEntity(const EntityPtr& e) {
        if (e->GetParentNode() != nullptr) {
            throw std::runtime_error("Unsupported operation - tree structure is immutable after adding to manager");
        }
        std::unique_lock<std::shared_mutex> ul(_m);
        _RemoveEntity(e);
    }

    void EntityManager::_AddEntity(const EntityPtr& e) {
        _entitiesToAdd.insert(e);
        for (const EntityPtr& c : e->GetChildNodes()) {
            _AddEntity(c);
        }
    }

    void EntityManager::_RemoveEntity(const EntityPtr& e) {
        _entitiesToRemove.insert(e);    
        for (const EntityPtr& c : e->GetChildNodes()) {
            _RemoveEntity(c);
        }
    }

    bool EntityManager::Initialize() {
        // Initialize core engine entity processors
        RegisterEntityProcess<TransformProcess>();

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

        for (auto ptr : entitiesToAdd) ptr->_AddToWorld();
        for (auto ptr : entitiesToRemove) ptr->_RemoveFromWorld();

        for (EntityProcessPtr& ptr : _processes) {
            if (entitiesToAdd.size() > 0) ptr->EntitiesAdded(entitiesToAdd);
            if (addedComponents.size() > 0) ptr->EntityComponentsAdded(addedComponents);
            if (entitiesToRemove.size() > 0) ptr->EntitiesRemoved(entitiesToRemove);
            if (componentsEnabledDisabled.size() > 0) ptr->EntityComponentsEnabledDisabled(componentsEnabledDisabled);
            ptr->Process(deltaSeconds);
        }

        // Commit added/removed entities
        for (auto& e : entitiesToAdd) _entities.insert(e);
        for (auto& e : entitiesToRemove) _entities.erase(e);

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
    
    void EntityManager::_NotifyComponentsAdded(const EntityPtr& ptr, EntityComponent * component) {
        std::unique_lock<std::shared_mutex> ul(_m);
        auto it = _addedComponents.find(ptr);
        if (it == _addedComponents.end()) {
            _addedComponents.insert(std::make_pair(ptr, std::vector<EntityComponent *>{component}));
        }
        else {
            it->second.push_back(component);
        }
    }

    void EntityManager::_NotifyComponentsEnabledDisabled(const EntityPtr& ptr) {
        std::unique_lock<std::shared_mutex> ul(_m);
        _componentsEnabledDisabled.insert(ptr);
    }
}