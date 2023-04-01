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
           return ThreadSafePoolAllocator<Entity>::AllocateSharedCustomConstruct(PlacementNew_<EntityComponentSet *>, ptr);
        }
        return ThreadSafePoolAllocator<Entity>::AllocateSharedCustomConstruct(PlacementNew_<>);
    }

    void EntityComponent::MarkChanged() {
        if (INSTANCE(Engine)) {
            lastFrameChanged_ = INSTANCE(Engine)->FrameCount();
        }
    }

    bool EntityComponent::ChangedLastFrame() const {
        uint64_t diff = INSTANCE(Engine)->FrameCount() - lastFrameChanged_;
        return diff == 1;
    }

    bool EntityComponent::ChangedThisFrame() const {
        uint64_t diff = INSTANCE(Engine)->FrameCount() - lastFrameChanged_;
        return diff == 0;
    }

    bool EntityComponent::ChangedWithinLastFrame() const {
        uint64_t diff = INSTANCE(Engine)->FrameCount() - lastFrameChanged_;
        return diff <= 1;
    }

    EntityComponentSet::~EntityComponentSet() {
        componentManagers_.clear();
        components_.clear();
        componentTypeNames_.clear();
    }

    void EntityComponentSet::SetOwner_(Entity * owner) {
        owner_ = owner;
    }

    Entity::Entity() : Entity(EntityComponentSet::Create()) {}

    Entity::Entity(EntityComponentSet * ptr) {
        handle_ = EntityHandle::NextHandle();
        components_ = ptr;
        components_->SetOwner_(this);
    }

    const EntityHandle& Entity::GetHandle() const {
        return handle_;
    }

    Entity::~Entity() {
        childNodes_.clear();
        EntityComponentSet::Destroy(components_);
    }

    bool Entity::IsInWorld() const {
        auto sl = std::shared_lock<std::shared_mutex>(m_);
        return partOfWorld_;
    }

    EntityComponentSet * EntityComponentSet::Copy() const {
        //auto sl = std::shared_lock<std::shared_mutex>(_m);
        EntityComponentSet * copy = EntityComponentSet::Create();
        for (const auto& manager : componentManagers_) {
            auto mgrCopy = CopyManager_(manager);
            copy->AttachComponent_(mgrCopy);
        }
        return copy;
    }

    void EntityComponentSet::AttachComponent_(std::unique_ptr<EntityComponentPointerManager>& ptr) {
        EntityComponentView view(ptr->component);
        const std::string name = view.component->TypeName();
        componentManagers_.push_back(std::move(ptr));
        components_.insert(view);
        componentTypeNames_.insert(std::make_pair(name, std::make_pair(view, EntityComponentStatus::COMPONENT_ENABLED)));

        if (owner_ && owner_->IsInWorld()) {
            INSTANCE(EntityManager)->NotifyComponentsAdded_(owner_->shared_from_this(), view.component);
        }
    }

    void EntityComponentSet::NotifyEntityManagerComponentEnabledDisabled_() {
        if (owner_ && owner_->IsInWorld()) {
            INSTANCE(EntityManager)->NotifyComponentsEnabledDisabled_(owner_->shared_from_this());
        }
    }

    EntityComponentPair<EntityComponent> EntityComponentSet::GetComponentByName(const std::string& name) {
        return GetComponentByName_<EntityComponent>(name);
    }

    EntityComponentPair<const EntityComponent> EntityComponentSet::GetComponentByName(const std::string& name) const {
        return GetComponentByName_<const EntityComponent>(name);
    }

    std::vector<EntityComponentPair<EntityComponent>> EntityComponentSet::GetAllComponents() {
        //auto sl = std::shared_lock<std::shared_mutex>(_m);
        std::vector<EntityComponentPair<EntityComponent>> v;
        v.reserve(components_.size());
        for (auto& component : componentTypeNames_) {
            v.push_back(EntityComponentPair<EntityComponent>{component.second.first.component, component.second.second});
        }
        return v;
    }

    std::vector<EntityComponentPair<const EntityComponent>> EntityComponentSet::GetAllComponents() const {
        //auto sl = std::shared_lock<std::shared_mutex>(_m);
        std::vector<EntityComponentPair<const EntityComponent>> v;
        v.reserve(components_.size());
        for (const auto& component : componentTypeNames_) {
            v.push_back(EntityComponentPair<const EntityComponent>{component.second.first.component, component.second.second});
        }
        return v;
    }

    EntityComponentSet& Entity::Components() {
        return *components_;
    }

    const EntityComponentSet& Entity::Components() const {
        return *components_;
    }

    // Called by World class
    void Entity::AddToWorld_() {
        partOfWorld_ = true;
    }

    void Entity::RemoveFromWorld_() {
        partOfWorld_ = false;
    }

    void Entity::AttachChildNode(const EntityPtr& ptr) {
        if (IsInWorld()) throw std::runtime_error("Entity is part of world - tree is immutable");
        auto self = shared_from_this();
        if (ptr == self) return;
        // If there is already a parent then don't attempt to overwrite
        if (ptr->GetParentNode() != nullptr) return;
        //auto ul = std::unique_lock<std::shared_mutex>(_m);
        if (ContainsChildNode_(ptr) || ptr->ContainsChildNode(self)) return;
        childNodes_.push_back(ptr);
        ptr->parent_ = self;
    }

    void Entity::DetachChildNode(const EntityPtr& ptr) {
        if (IsInWorld()) throw std::runtime_error("Entity is part of world - tree is immutable");
        std::vector<EntityPtr> visited;
        {
            //auto ul = std::unique_lock<std::shared_mutex>(_m);
            for (auto it = childNodes_.begin(); it != childNodes_.end(); ++it) {
                EntityPtr c = *it;
                if (c == ptr) {
                    childNodes_.erase(it);
                    c->parent_.reset();
                    return;
                }
                visited.push_back(c);
            }
        }
        // If we get here we still have to try removing the node further down
        for (EntityPtr v : visited) v->DetachChildNode(ptr);
    }

    EntityPtr Entity::GetParentNode() const {
        return parent_.lock();
    }

    const std::vector<EntityPtr>& Entity::GetChildNodes() const {
        return childNodes_;
    }

    bool Entity::ContainsChildNode(const EntityPtr& ptr) const {
        auto sl = std::shared_lock<std::shared_mutex>(m_);
        return ContainsChildNode_(ptr);
    }

    bool Entity::ContainsChildNode_(const EntityPtr& ptr) const {
        for (const EntityPtr& c : childNodes_) {
            if (c == ptr) return true;
            const bool nestedCheck = c->ContainsChildNode_(ptr);
            if (nestedCheck) return true;
        }
        return false;
    }

    EntityPtr Entity::Copy() const {
        auto sl = std::shared_lock<std::shared_mutex>(m_);
        auto components = components_->Copy();
        auto copy = Entity::Create(components);
        for (auto& ptr : childNodes_) {
            auto child = ptr->Copy();
            copy->childNodes_.push_back(child);
            child->parent_ = copy;
        }
        return copy;
    }
}