#include "StratusEntity.h"
#include "StratusEntityManager.h"
#include "StratusEngine.h"
#include "StratusPoolAllocator.h"

namespace stratus {
    EntityPtr Entity::Create() {
        return Create(nullptr);
        //return EntityPtr(new Entity());
    }

    EntityPtr Entity::Create(EntityComponentSet * ptr) {
        if (ptr != nullptr) {
           return ThreadSafePoolAllocator<Entity>::AllocateSharedCustomConstruct(_PlacementNew<EntityComponentSet *>, ptr);
        }
        return ThreadSafePoolAllocator<Entity>::AllocateSharedCustomConstruct(_PlacementNew<>);
    }

    void EntityComponent::MarkChanged() {
        if (INSTANCE(Engine)) {
            _lastFrameChanged = INSTANCE(Engine)->FrameCount();
        }
    }

    bool EntityComponent::ChangedLastFrame() const {
        uint64_t diff = INSTANCE(Engine)->FrameCount() - _lastFrameChanged;
        return diff == 1;
    }

    bool EntityComponent::ChangedThisFrame() const {
        uint64_t diff = INSTANCE(Engine)->FrameCount() - _lastFrameChanged;
        return diff == 0;
    }

    bool EntityComponent::ChangedWithinLastFrame() const {
        uint64_t diff = INSTANCE(Engine)->FrameCount() - _lastFrameChanged;
        return diff <= 1;
    }

    EntityComponentSet::~EntityComponentSet() {
        _componentManagers.clear();
        _components.clear();
        _componentTypeNames.clear();
    }

    void EntityComponentSet::_SetOwner(Entity * owner) {
        _owner = owner;
    }

    Entity::Entity() : Entity(EntityComponentSet::Create()) {}

    Entity::Entity(EntityComponentSet * ptr) {
        _components = ptr;
        _components->_SetOwner(this);
    }

    Entity::~Entity() {
        _childNodes.clear();
        EntityComponentSet::Destroy(_components);
    }

    bool Entity::IsInWorld() const {
        auto sl = std::shared_lock<std::shared_mutex>(_m);
        return _partOfWorld;
    }

    EntityComponentSet * EntityComponentSet::Copy() const {
        //auto sl = std::shared_lock<std::shared_mutex>(_m);
        EntityComponentSet * copy = EntityComponentSet::Create();
        for (const auto& manager : _componentManagers) {
            auto mgrCopy = __CopyManager(manager);
            copy->_AttachComponent(mgrCopy);
        }
        return copy;
    }

    void EntityComponentSet::_AttachComponent(std::unique_ptr<EntityComponentPointerManager>& ptr) {
        EntityComponentView view(ptr->component);
        const std::string name = view.component->TypeName();
        _componentManagers.push_back(std::move(ptr));
        _components.insert(view);
        _componentTypeNames.insert(std::make_pair(name, std::make_pair(view, EntityComponentStatus::COMPONENT_ENABLED)));

        if (_owner && _owner->IsInWorld()) {
            INSTANCE(EntityManager)->_NotifyComponentsAdded(_owner->shared_from_this(), view.component);
        }
    }

    void EntityComponentSet::_NotifyEntityManagerComponentEnabledDisabled() {
        if (_owner && _owner->IsInWorld()) {
            INSTANCE(EntityManager)->_NotifyComponentsEnabledDisabled(_owner->shared_from_this());
        }
    }

    EntityComponentPair<EntityComponent> EntityComponentSet::GetComponentByName(const std::string& name) {
        return _GetComponentByName<EntityComponent>(name);
    }

    EntityComponentPair<const EntityComponent> EntityComponentSet::GetComponentByName(const std::string& name) const {
        return _GetComponentByName<const EntityComponent>(name);
    }

    std::vector<EntityComponentPair<EntityComponent>> EntityComponentSet::GetAllComponents() {
        //auto sl = std::shared_lock<std::shared_mutex>(_m);
        std::vector<EntityComponentPair<EntityComponent>> v;
        v.reserve(_components.size());
        for (auto& component : _componentTypeNames) {
            v.push_back(EntityComponentPair<EntityComponent>{component.second.first.component, component.second.second});
        }
        return v;
    }

    std::vector<EntityComponentPair<const EntityComponent>> EntityComponentSet::GetAllComponents() const {
        //auto sl = std::shared_lock<std::shared_mutex>(_m);
        std::vector<EntityComponentPair<const EntityComponent>> v;
        v.reserve(_components.size());
        for (const auto& component : _componentTypeNames) {
            v.push_back(EntityComponentPair<const EntityComponent>{component.second.first.component, component.second.second});
        }
        return v;
    }

    EntityComponentSet& Entity::Components() {
        return *_components;
    }

    const EntityComponentSet& Entity::Components() const {
        return *_components;
    }

    // Called by World class
    void Entity::_AddToWorld() {
        _partOfWorld = true;
    }

    void Entity::_RemoveFromWorld() {
        _partOfWorld = false;
    }

    void Entity::AttachChildNode(const EntityPtr& ptr) {
        if (IsInWorld()) throw std::runtime_error("Entity is part of world - tree is immutable");
        auto self = shared_from_this();
        if (ptr == self) return;
        // If there is already a parent then don't attempt to overwrite
        if (ptr->GetParentNode() != nullptr) return;
        //auto ul = std::unique_lock<std::shared_mutex>(_m);
        if (_ContainsChildNode(ptr) || ptr->ContainsChildNode(self)) return;
        _childNodes.push_back(ptr);
        ptr->_parent = self;
    }

    void Entity::DetachChildNode(const EntityPtr& ptr) {
        if (IsInWorld()) throw std::runtime_error("Entity is part of world - tree is immutable");
        std::vector<EntityPtr> visited;
        {
            //auto ul = std::unique_lock<std::shared_mutex>(_m);
            for (auto it = _childNodes.begin(); it != _childNodes.end(); ++it) {
                EntityPtr c = *it;
                if (c == ptr) {
                    _childNodes.erase(it);
                    c->_parent.reset();
                    return;
                }
                visited.push_back(c);
            }
        }
        // If we get here we still have to try removing the node further down
        for (EntityPtr v : visited) v->DetachChildNode(ptr);
    }

    EntityPtr Entity::GetParentNode() const {
        return _parent.lock();
    }

    const std::vector<EntityPtr>& Entity::GetChildNodes() const {
        return _childNodes;
    }

    bool Entity::ContainsChildNode(const EntityPtr& ptr) const {
        auto sl = std::shared_lock<std::shared_mutex>(_m);
        return _ContainsChildNode(ptr);
    }

    bool Entity::_ContainsChildNode(const EntityPtr& ptr) const {
        for (const EntityPtr& c : _childNodes) {
            if (c == ptr) return true;
            const bool nestedCheck = c->_ContainsChildNode(ptr);
            if (nestedCheck) return true;
        }
        return false;
    }

    EntityPtr Entity::Copy() const {
        auto sl = std::shared_lock<std::shared_mutex>(_m);
        auto components = _components->Copy();
        auto copy = Entity::Create(components);
        for (auto& ptr : _childNodes) {
            auto child = ptr->Copy();
            copy->_childNodes.push_back(child);
            child->_parent = copy;
        }
        return copy;
    }
}