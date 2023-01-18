#include "StratusEntity.h"
#include "StratusEngine.h"
#include <glm/gtx/matrix_decompose.hpp>

namespace stratus {
    EntityPtr Entity::Create() {
        return EntityPtr(new Entity());
    }

    Entity::Entity() {
        _handle = EntityHandle::NextHandle();
        _refCount = std::make_shared<std::atomic<uint64_t>>(1);
        _RecalcTransform();
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

    glm::vec3 Entity::GetLocalPosition() const {
        return glm::vec3(_localTranslate[3].x, _localTranslate[3].y, _localTranslate[3].z);
    }

    void Entity::SetLocalPosition(const glm::vec3& pos) {
        matTranslate(_localTranslate, pos);
        _RecalcTransform();
    }

    const glm::mat4& Entity::GetLocalRotation() const {
        return _localRotate;
    }

    void Entity::SetLocalRotation(const Rotation& rot) {
        SetLocalRotation(rot.asMat4());
    }

    void Entity::SetLocalRotation(const glm::mat4& rot) {
        _localRotate = rot;
        _RecalcTransform();
    }

    glm::vec3 Entity::GetLocalScale() const {
        return glm::vec3(_localScale[0].x, _localScale[1].y, _localScale[2].z);
    }

    void Entity::SetLocalScale(const glm::vec3& scale) {
        glm::mat4 m(1.0f);
        matScale(m, scale);
        _localScale = std::move(m);
        _RecalcTransform();
    }

    void Entity::SetLocalPosRotScale(const glm::vec3& pos, const Rotation& rot, const glm::vec3& scale) {
        SetLocalPosRotScale(pos, rot.asMat4(), scale);
    }

    void Entity::SetLocalPosRotScale(const glm::vec3& pos, const glm::mat4& rot, const glm::vec3& scale) {
        matTranslate(_localTranslate, pos);
        _localRotate = rot;
        _localScale = glm::mat4(1.0f);
        matScale(_localScale, scale);
        _RecalcTransform();
    }

    glm::vec3 Entity::GetWorldPosition() const {
        return glm::vec3(_worldTransform[3].x, _worldTransform[3].y, _worldTransform[3].z);
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
        _localTransform = _localTranslate * _localRotate * _localScale;
        _worldTransform = _parent != nullptr ? _parent->_worldTransform * _localTransform : _localTransform;

        if (_renderNode != nullptr) {
            _renderNode->SetGlobalTransform(_worldTransform);
        }

        for (auto& child : _children) {
            child->_RecalcTransform();
        }

        _lastFrameChanged = Engine::Instance()->FrameCount();
    }

    bool Entity::ChangedWithinLastFrame() const {
        auto offset = Engine::Instance()->FrameCount() - _lastFrameChanged;
        return offset < 2;
    }

    RenderNodePtr Entity::GetRenderNode() const {
        return _renderNode;
    }

    void Entity::SetRenderNode(const RenderNodePtr& node) {
        _renderNode = node;
        _renderNode->SetGlobalTransform(_worldTransform);
        _renderNode->SetOwner(shared_from_this());
    }

    EntityPtr Entity::Copy(bool copyHandle) const {
        auto entity = Create();
        Copy(entity, copyHandle);
        return entity;
    }

    void Entity::Copy(EntityPtr& ptr, bool copyHandle) const {
        if (copyHandle) {
            ptr->_handle = _handle;
        }
        else {
            ptr->_handle = EntityHandle::NextHandle();
        }
        ptr->_refCount = _refCount;
        ptr->IncrRefCount();

        ptr->_localTranslate = _localTranslate;
        ptr->_localScale = _localScale;
        ptr->_localRotate = _localRotate;
        ptr->_localTransform = GetLocalTransform();
        ptr->_worldTransform = GetWorldTransform();

        if (_renderNode != nullptr) {
            ptr->_renderNode = _renderNode->Copy();
        }

        for (auto& child : _children) {
            auto childCopy = child->Copy(copyHandle);
            childCopy->_parent = ptr;
            ptr->_children.insert(childCopy);
        }
    }
}