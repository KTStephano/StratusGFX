#include "StratusEntity.h"

    //     Entity();
    //     ~Entity();

    //     // Functions for manipulating entity components
    //     void AddComponent(const EntityComponentPtr&);
    //     void RemoveComponent(const EntityComponentPtr&);
    //     void RemoveComponentByHandle(const EntityComponentHandle&);
    //     void RemoveAllComponents();
    //     const std::vector<EntityComponentPtr>& GetComponents() const;

    //     // Functions for getting and setting transform
    //     const glm::vec3& GetPosition() const;
    //     void SetPosition(const glm::vec3&);

    //     const Rotation& GetRotation() const;
    //     void SetRotation(const Rotation&);

    //     const glm::vec3& GetScale() const;
    //     void SetScale(const glm::vec3&);

    //     void SetPosRotScale(const glm::vec3&, const Rotation&, const glm::vec3&);

    //     // World transform is relative to the entity's parent
    //     const glm::mat4& GetWorldTransform() const;
    //     // Local transform is only taking into account this current entity
    //     const glm::mat4& GetLocalTransform() const;

    //     // Shows which events this entity needs to receive
    //     uint64_t GetEventMask() const;
    //     // Called by the scene for each event that occurs
    //     void SendEvent(const EntityEvent&);
    //     // Queueing and getting queued events
    //     void QueueEvent(const EntityEvent&);
    //     void GetQueuedEvents(std::vector<EntityEvent>& out);
    //     // Clears all events (happens after all events have been processed)
    //     void ClearQueuedEvents();

    //     // Functions for dealing with parent and child entities
    //     void SetParent(EntityPtr);
    //     EntityPtr GetParent() const;

    //     void AttachChild(EntityPtr);
    //     void DetachChild(EntityPtr);
    //     const std::vector<EntityPtr>& GetChildren() const;

    //     EntityHandle GetHandle() const;

    //     // Functions for setting (optional) entity name and getting it
    //     void SetName(const std::string&);
    //     const std::string& GetName() const;

    //     // Creates a deep copy of this entity, its components, and all other nodes
    //     EntityPtr Copy() const;

    // private:
    //     EntityPtr _parent;
    //     std::vector<EntityPtr> _children;
    //     RenderNodePtr _renderNode;
        // glm::vec3 _position;
        // Rotation _rotation;
        // glm::vec3 _scale;
        // glm::mat4 _localTransform = glm::mat4(1.0f);
        // glm::mat4 _worldTransform = glm::mat4(1.0f);

namespace stratus {
    EntityPtr Entity::Create() {
        return EntityPtr(new Entity());
    }

    Entity::Entity() {
        _handle = EntityHandle::NextHandle();
        _refCount = std::make_shared<std::atomic<uint64_t>>(1);
    }

    uint64_t Entity::GetRefCount() const {
        return _refCount->load();
    }

    void Entity::IncrRefCount() {
        _refCount->fetch_add(1);
    }

    void Entity::DecrRefCount() {
        _refCount->fetch_sub(1);
    }

    Entity::~Entity() {
        DecrRefCount();
    }

    void Entity::AddComponent(const EntityComponentPtr&) {
        throw std::runtime_error("Implement");
    }

    void Entity::RemoveComponent(const EntityComponentPtr&) {
        throw std::runtime_error("Implement");

    }

    void Entity::RemoveComponentByHandle(const EntityComponentHandle&) {
        throw std::runtime_error("Implement");

    }

    void Entity::RemoveAllComponents() {
        throw std::runtime_error("Implement");

    }

    const std::vector<EntityComponentPtr>& Entity::GetComponents() const {
        throw std::runtime_error("Implement");

    }

    const glm::vec3& Entity::GetPosition() const {
        return _position;
    }

    void Entity::SetPosition(const glm::vec3& pos) {
        _position = pos;
        _RecalcTransform();
    }

    const Rotation& Entity::GetRotation() const {
        return _rotation;
    }

    void Entity::SetRotation(const Rotation& rot) {
        _rotation = rot;
        _RecalcTransform();
    }

    const glm::vec3& Entity::GetScale() const {
        return _scale;
    }

    void Entity::SetScale(const glm::vec3& scale) {
        _scale = scale;
        _RecalcTransform();
    }

    void Entity::SetPosRotScale(const glm::vec3& pos, const Rotation& rot, const glm::vec3& scale) {
        _position = pos;
        _rotation = rot;
        _scale = scale;
        _RecalcTransform();
    }

    const glm::mat4& Entity::GetWorldTransform() const {
        return _worldTransform;
    }

    const glm::mat4& Entity::GetLocalTransform() const {
        return _localTransform;
    }

    uint64_t Entity::GetEventMask() const {
        throw std::runtime_error("Implement");

    }

    void Entity::SendEvent(const EntityEvent&) {
        throw std::runtime_error("Implement");
    }

    void Entity::QueueEvent(const EntityEvent&) {
        throw std::runtime_error("Implement");

    }

    void Entity::GetQueuedEvents(std::vector<EntityEvent>& out) {
        throw std::runtime_error("Implement");

    }

    void Entity::ClearQueuedEvents() {
        throw std::runtime_error("Implement");
    }

    void Entity::_SetParent(EntityPtr parent) {
        _parent = parent;
    }

    EntityPtr Entity::GetParent() const {
        return _parent;
    }

    void Entity::AttachChild(EntityPtr child) {
        auto sharedThis = shared_from_this();
        if (child->GetParent() == sharedThis) return;
        else if (child->GetParent() != nullptr) {
            child->GetParent()->DetachChild(child);
        }

        _children.insert(child);
        child->_SetParent(sharedThis);
        child->_RecalcTransform();
    }

    void Entity::DetachChild(EntityPtr child) {
        if (_children.find(child) == _children.end()) return;
        _children.erase(child);
        child->_SetParent(nullptr);
    }

    const std::unordered_set<EntityPtr>& Entity::GetChildren() const {
        return _children;
    }

    EntityHandle Entity::GetHandle() const {
        return _handle;
    }

    void Entity::SetName(const std::string&) {
        throw std::runtime_error("Implement");
    }

    const std::string& Entity::GetName() const {
        throw std::runtime_error("Implement");
    }

    void Entity::_RecalcTransform() {
        _localTransform = constructTransformMat(_rotation, _position, _scale);
        _worldTransform = _parent != nullptr ? _parent->_worldTransform : glm::mat4(1.0f);
        _worldTransform = _worldTransform * _localTransform;
        
        if (_renderNode != nullptr) {
            _renderNode->SetWorldTransform(_worldTransform);
        }

        for (auto& child : _children) {
            child->_RecalcTransform();
        }
    }

    RenderNodePtr Entity::GetRenderNode() const {
        return _renderNode;
    }

    void Entity::SetRenderNode(const RenderNodePtr& node) {
        _renderNode = node;
        _renderNode->SetWorldTransform(_worldTransform);
    }

    EntityPtr Entity::Copy() const {
        auto entity = Create();
        Copy(entity);
        return entity;
    }

    void Entity::Copy(EntityPtr& ptr) const {
        ptr->_handle = _handle;
        ptr->_refCount = _refCount;
        ptr->IncrRefCount();

        ptr->_position = GetPosition();
        ptr->_rotation = GetRotation();
        ptr->_scale = GetScale();
        ptr->_localTransform = GetLocalTransform();
        ptr->_worldTransform = GetWorldTransform();

        if (_renderNode != nullptr) {
            ptr->_renderNode = _renderNode->Copy();
        }

        for (auto& child : _children) {
            auto childCopy = child->Copy();
            childCopy->_parent = ptr;
            ptr->_children.insert(childCopy);
        }
    }
}