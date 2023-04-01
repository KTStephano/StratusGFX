#include "StratusEntityManager.h"
#include "StratusEntity.h"
#include "StratusApplicationThread.h"
#include "StratusTransformComponent.h"
#include <algorithm>

namespace stratus {
    EntityManager::EntityManager() {}

    void EntityManager::AddEntity(const EntityPtr& e) {
        if (e == nullptr) return;
        if (e->GetParentNode() != nullptr) {
            throw std::runtime_error("Unsupported operation - must add root node");
        }
        std::unique_lock<std::shared_mutex> ul(m_);
        AddEntity_(e);
    }

    void EntityManager::RemoveEntity(const EntityPtr& e) {
        if (e == nullptr) return;
        if (e->GetParentNode() != nullptr) {
            throw std::runtime_error("Unsupported operation - tree structure is immutable after adding to manager");
        }
        std::unique_lock<std::shared_mutex> ul(m_);
        RemoveEntity_(e);
    }

    void EntityManager::AddEntity_(const EntityPtr& e) {
        entitiesToAdd_.insert(e);
        for (const EntityPtr& c : e->GetChildNodes()) {
            AddEntity_(c);
        }
    }

    void EntityManager::RemoveEntity_(const EntityPtr& e) {
        entitiesToRemove_.insert(e);    
        for (const EntityPtr& c : e->GetChildNodes()) {
            RemoveEntity_(c);
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
        auto entitiesToAdd = std::move(entitiesToAdd_);
        auto addedComponents = std::move(addedComponents_);
        auto entitiesToRemove = std::move(entitiesToRemove_);
        auto componentsEnabledDisabled = std::move(componentsEnabledDisabled_);

        for (auto ptr : entitiesToAdd) ptr->AddToWorld_();
        for (auto ptr : entitiesToRemove) ptr->RemoveFromWorld_();

        for (EntityProcessPtr& ptr : processes_) {
            if (entitiesToAdd.size() > 0) ptr->EntitiesAdded(entitiesToAdd);
            if (addedComponents.size() > 0) ptr->EntityComponentsAdded(addedComponents);
            if (entitiesToRemove.size() > 0) ptr->EntitiesRemoved(entitiesToRemove);
            if (componentsEnabledDisabled.size() > 0) ptr->EntityComponentsEnabledDisabled(componentsEnabledDisabled);
            ptr->Process(deltaSeconds);
        }

        // Commit added/removed entities
        for (auto& e : entitiesToAdd) entities_.insert(e);
        for (auto& e : entitiesToRemove) entities_.erase(e);

        // If any processes have been added, tell them about all available entities
        // and allow them to perform their process routine for the first time
        auto processesToAdd = std::move(processesToAdd_);
        for (EntityProcessPtr& ptr : processesToAdd) {
            if (entities_.size() > 0) ptr->EntitiesAdded(entities_);
            ptr->Process(deltaSeconds);

            // Commit process to list
            processes_.push_back(ptr);
            handlesToPtrs_.insert(std::make_pair((EntityProcessHandle)ptr.get(), ptr));
        }

        // If any processes have been removed then remove them now
        auto processesToRemove = std::move(processesToRemove_);
        for (EntityProcessHandle handle : processesToRemove) {
            auto handleIt = handlesToPtrs_.find(handle);
            if (handleIt == handlesToPtrs_.end()) continue;
            auto remove = handleIt->second;
            for (auto it = processes_.begin(); it != processes_.end(); ++it) {
                EntityProcessPtr process = *it;
                if (process == remove) {
                    processes_.erase(it);
                    break;
                }
            }
            handlesToPtrs_.erase(handle);
        }

        return SystemStatus::SYSTEM_CONTINUE;
    }
    
    void EntityManager::Shutdown() {
        entities_.clear();
        entitiesToAdd_.clear();
        entitiesToRemove_.clear();
        processes_.clear();
        processesToAdd_.clear();
        addedComponents_.clear();
    }
    
    void EntityManager::RegisterEntityProcess_(EntityProcessPtr& ptr) {
        std::unique_lock<std::shared_mutex> ul(m_);
        processesToAdd_.push_back(std::move(ptr));
    }

    void EntityManager::UnregisterEntityProcess(EntityProcessHandle handle) {
        std::unique_lock<std::shared_mutex> ul(m_);
        processesToRemove_.insert(handle);
    }
    
    void EntityManager::NotifyComponentsAdded_(const EntityPtr& ptr, EntityComponent * component) {
        std::unique_lock<std::shared_mutex> ul(m_);
        auto it = addedComponents_.find(ptr);
        if (it == addedComponents_.end()) {
            addedComponents_.insert(std::make_pair(ptr, std::vector<EntityComponent *>{component}));
        }
        else {
            it->second.push_back(component);
        }
    }

    void EntityManager::NotifyComponentsEnabledDisabled_(const EntityPtr& ptr) {
        std::unique_lock<std::shared_mutex> ul(m_);
        componentsEnabledDisabled_.insert(ptr);
    }
}