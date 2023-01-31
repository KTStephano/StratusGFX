#include "StratusEntity2.h"
#include "StratusEntityManager.h"
#include "StratusEngine.h"
#include "StratusPoolAllocator.h"

namespace stratus {
    Entity2Ptr Entity2::Create() {
        static std::mutex m;
        static PoolAllocator<Entity2> allocator;

        struct Deleter {
            void operator()(Entity2 * ptr) {
                std::unique_lock<std::mutex> ul(m);
                allocator.Deallocate(ptr);
            }
        };

        std::unique_lock<std::mutex> ul(m);
        return Entity2Ptr(allocator.AllocateCustomConstruct(_PlacementNew<>), Deleter());
    }

    void Entity2Component::MarkChanged() {
        if (INSTANCE(Engine)) {
            _lastFrameChanged = INSTANCE(Engine)->FrameCount();
        }
    }

    bool Entity2Component::ChangedWithinLastFrame() const {
        uint64_t diff = INSTANCE(Engine)->FrameCount() - _lastFrameChanged;
        return diff <= 1;
    }

    Entity2ComponentSet::~Entity2ComponentSet() {
        _componentManagers.clear();
        _components.clear();
        _componentTypeNames.clear();
    }

    void Entity2ComponentSet::_SetOwner(Entity2 * owner) {
        _owner = owner;
    }

    Entity2::Entity2() {
        _components._SetOwner(this);
    }

    Entity2::~Entity2() {
        _childNodes.clear();
    }

    bool Entity2::IsInWorld() const {
        auto sl = std::shared_lock<std::shared_mutex>(_m);
        return _partOfWorld;
    }

    void Entity2ComponentSet::_AttachComponent(Entity2ComponentView view) {
        const std::string name = view.component->TypeName();
        _components.insert(view);
        _componentTypeNames.insert(std::make_pair(name, std::make_pair(view, EntityComponentStatus::COMPONENT_ENABLED)));
        INSTANCE(EntityManager)->_NotifyComponentsAdded(_owner->shared_from_this(), view.component);
    }

    void Entity2ComponentSet::_NotifyEntityManagerComponentEnabledDisabled() {
        INSTANCE(EntityManager)->_NotifyComponentsEnabledDisabled(_owner->shared_from_this());
    }

    std::vector<EntityComponentPair<Entity2Component>> Entity2ComponentSet::GetAllComponents() {
        auto sl = std::shared_lock<std::shared_mutex>(_m);
        std::vector<EntityComponentPair<Entity2Component>> v;
        v.reserve(_components.size());
        for (auto& component : _componentTypeNames) {
            v.push_back(EntityComponentPair<Entity2Component>{component.second.first.component, component.second.second});
        }
        return v;
    }

    std::vector<EntityComponentPair<const Entity2Component>> Entity2ComponentSet::GetAllComponents() const {
        auto sl = std::shared_lock<std::shared_mutex>(_m);
        std::vector<EntityComponentPair<const Entity2Component>> v;
        v.reserve(_components.size());
        for (const auto& component : _componentTypeNames) {
            v.push_back(EntityComponentPair<const Entity2Component>{component.second.first.component, component.second.second});
        }
        return v;
    }

    Entity2ComponentSet& Entity2::Components() {
        return _components;
    }

    const Entity2ComponentSet& Entity2::Components() const {
        return _components;
    }

    // Called by World class
    void Entity2::_AddToWorld() {
        _partOfWorld = true;
    }

    void Entity2::_RemoveFromWorld() {
        _partOfWorld = false;
    }

    void Entity2::AttachChildNode(const Entity2Ptr& ptr) {
        if (IsInWorld()) throw std::runtime_error("Entity2 is part of world - tree is immutable");
        auto self = shared_from_this();
        if (ptr == self) return;
        // If there is already a parent then don't attempt to overwrite
        if (GetParentNode() != nullptr) return;
        //auto ul = std::unique_lock<std::shared_mutex>(_m);
        if (_ContainsChildNode(ptr) || ptr->ContainsChildNode(self)) return;
        _childNodes.push_back(ptr);
        ptr->_parent = self;
    }

    void Entity2::DetachChildNode(const Entity2Ptr& ptr) {
        if (IsInWorld()) throw std::runtime_error("Entity2 is part of world - tree is immutable");
        std::vector<Entity2Ptr> visited;
        {
            //auto ul = std::unique_lock<std::shared_mutex>(_m);
            for (auto it = _childNodes.begin(); it != _childNodes.end(); ++it) {
                Entity2Ptr c = *it;
                if (c == ptr) {
                    _childNodes.erase(it);
                    c->_parent.reset();
                    return;
                }
                visited.push_back(c);
            }
        }
        // If we get here we still have to try removing the node further down
        for (Entity2Ptr v : visited) v->DetachChildNode(ptr);
    }

    Entity2Ptr Entity2::GetParentNode() const {
        return _parent.lock();
    }

    const std::vector<Entity2Ptr>& Entity2::GetChildNodes() const {
        return _childNodes;
    }

    bool Entity2::ContainsChildNode(const Entity2Ptr& ptr) const {
        auto sl = std::shared_lock<std::shared_mutex>(_m);
        return _ContainsChildNode(ptr);
    }

    bool Entity2::_ContainsChildNode(const Entity2Ptr& ptr) const {
        for (const Entity2Ptr& c : _childNodes) {
            if (c == ptr) return true;
            const bool nestedCheck = c->_ContainsChildNode(ptr);
            if (nestedCheck) return true;
        }
        return false;
    }
}