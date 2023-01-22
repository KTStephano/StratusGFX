#include "StratusEntity2.h"
#include "StratusEntityManager.h"

namespace stratus {
    Entity2::Entity2() {

    }

    Entity2::~Entity2() {

    }

    bool Entity2::IsInWorld() const {
        return _partOfWorld;
    }

    // Called by World class
    void Entity2::_AddToWorld() {

    }

    void Entity2::_RemoveFromWorld() {

    }

    void Entity2::_CommitChanges() {
        // Clear pending remove
        for (Entity2ComponentView view : _pendingRemove) {
            _RemoveComponentImmediate(view);
        }
        _pendingRemove.clear();
    }

    void Entity2::_AttachComponent(Entity2ComponentView view) {
        const std::string name = view.component->TypeInfo().name();
        _components.insert(view);
        _componentTypeNames.insert(std::make_pair(name, view));
        if (IsInWorld()) {
            EntityManager::Instance()->_NotifyComponentsAdded(shared_from_this(), name);
        }
    }

    void Entity2::_RemoveComponent(Entity2ComponentView view) {
        if (IsInWorld()) {
            const std::string name = view.component->TypeInfo().name();
            _pendingRemove.insert(view);
            EntityManager::Instance()->_NotifyComponentsRemoved(shared_from_this(), name);
        }
        else {
            _RemoveComponentImmediate(view);
        }
    }

    void Entity2::_RemoveComponentImmediate(Entity2ComponentView view) {
        const std::string name = view.component->TypeInfo().name();
        _components.erase(view);
        _componentTypeNames.erase(name);
        delete view.component;
        view.component = nullptr;
    }
}