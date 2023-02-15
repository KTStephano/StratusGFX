#include "StratusTransformComponent.h"

namespace stratus {
    EntityPtr CreateTransformEntity() {
        auto ptr = Entity::Create();
        InitializeTransformEntity(ptr);
        return ptr;
    }

    void InitializeTransformEntity(EntityPtr ptr) {
        ptr->Components().AttachComponent<LocalTransformComponent>();
        ptr->Components().AttachComponent<GlobalTransformComponent>();
    }

    void LocalTransformComponent::SetLocalScale(const glm::vec3& scale) {
        SetLocalTransform(scale, _rotation, _position);
    }

    const glm::vec3& LocalTransformComponent::GetLocalScale() const {
        return _scale;
    }

    void LocalTransformComponent::SetLocalRotation(const Rotation& rot) {
        SetLocalRotation(rot.asMat3());
    }

    void LocalTransformComponent::SetLocalRotation(const glm::mat3& m) {
        SetLocalTransform(_scale, m, _position);
    }

    const glm::mat3& LocalTransformComponent::GetLocalRotation() const {
        return _rotation;
    }

    void LocalTransformComponent::SetLocalPosition(const glm::vec3& position) {
        SetLocalTransform(_scale, _rotation, position);
    }

    const glm::vec3& LocalTransformComponent::GetLocalPosition() const {
        return _position;
    }

    void LocalTransformComponent::SetLocalTransform(const glm::vec3& scale, const Rotation& rot, const glm::vec3& position) {
        SetLocalTransform(scale, rot.asMat3(), position);
    }

    void LocalTransformComponent::SetLocalTransform(const glm::vec3& scale, const glm::mat3& rot, const glm::vec3& position) {
        if (&scale != &_scale) _scale = scale;
        if (&rot != &_rotation) _rotation = rot;
        if (&position != &_position) _position = position;
        _MarkChangedAndRecalculate();
    }

    const glm::mat4& LocalTransformComponent::GetLocalTransform() const {
        return _transform;
    }

    void LocalTransformComponent::_MarkChangedAndRecalculate() {
        this->MarkChanged();
        auto S = glm::mat4(1.0f);
        matScale(S, _scale);
        auto R = glm::mat4(1.0f);
        matInset(R, _rotation); 
        auto T = glm::mat4(1.0f);
        matTranslate(T, _position);
        _transform = T * R * S;
        /*
        glm::mat3 RS = _rotation;
        matScale(RS, _scale);
        _transform = glm::mat4(1.0f);
        matTranslate(_transform, _position);
        matInset(_transform, RS);
        */
    }

    const glm::mat4& GlobalTransformComponent::GetGlobalTransform() const {
        return _transform;
    }

    void GlobalTransformComponent::_SetGlobalTransform(const glm::mat4& m) {
        this->MarkChanged();
        _transform = m;
    }

    TransformProcess::~TransformProcess() {}

    void TransformProcess::Process(const double deltaSeconds) {
        for (auto root : _rootNodes) {
            _ProcessNode(root);
        }
    }

    void TransformProcess::EntitiesAdded(const std::unordered_set<stratus::EntityPtr>& entities) {
        for (auto ptr : entities) {
            if (_IsEntityRelevant(ptr)) {
                std::vector<EntityComponent *> components{
                    ptr->Components().GetComponent<LocalTransformComponent>().component,
                    ptr->Components().GetComponent<GlobalTransformComponent>().component
                };
                
                _components.insert(std::make_pair(ptr, std::move(components)));

                if (ptr->GetParentNode() == nullptr) {
                    _rootNodes.insert(ptr);
                    _ProcessNode(ptr);
                }
            }
        }
    }

    void TransformProcess::EntitiesRemoved(const std::unordered_set<stratus::EntityPtr>& entities) {
        for (auto ptr : entities) {
            _rootNodes.erase(ptr);
            _components.erase(ptr);
        }
    }

    void TransformProcess::EntityComponentsAdded(const std::unordered_map<stratus::EntityPtr, std::vector<stratus::EntityComponent *>>& entities) {
        std::unordered_set<EntityPtr> added;
        for (auto p : entities) {
            if (_IsEntityRelevant(p.first)) {
                added.insert(p.first);
            }
        }

        EntitiesAdded(added);
    }

    void TransformProcess::EntityComponentsEnabledDisabled(const std::unordered_set<stratus::EntityPtr>& entities) {
        std::unordered_set<EntityPtr> added;
        std::unordered_set<EntityPtr> removed;
        for (auto ptr : entities) {
            if (_IsEntityRelevant(ptr)) {
                added.insert(ptr);
            }
            else {
                removed.insert(ptr);
            }
        }

        EntitiesAdded(added);
        EntitiesRemoved(removed);
    }

    bool TransformProcess::_IsEntityRelevant(const EntityPtr& e) {
        auto& components = e->Components();
        auto local = components.GetComponent<LocalTransformComponent>();
        auto global = components.GetComponent<GlobalTransformComponent>();
        if (local.component == nullptr || global.component == nullptr) return false;
        return local.status == EntityComponentStatus::COMPONENT_ENABLED &&
            global.status == EntityComponentStatus::COMPONENT_ENABLED;
    }

    void TransformProcess::_ProcessNode(const EntityPtr& p) {
        auto it = _components.find(p);
        if (it == _components.end()) return;

        bool parentChanged = false;
        auto parent = p->GetParentNode();
        GlobalTransformComponent * parentGlobal = nullptr;
        if (parent != nullptr) {
            auto itparent = _components.find(parent);
            if (itparent != _components.end()) {
                auto local = (LocalTransformComponent *)itparent->second[0];
                auto global = (GlobalTransformComponent *)itparent->second[1];
                parentChanged = local->ChangedLastFrame() || global->ChangedThisFrame();
                parentGlobal = global;
            }
        }

        auto local = (LocalTransformComponent *)it->second[0];
        auto global = (GlobalTransformComponent *)it->second[1];

        // See if local or parent changed requiring recompute of global transform
        if (parentChanged || local->ChangedLastFrame()) {
            global->_SetGlobalTransform(
                (parentGlobal ? parentGlobal->GetGlobalTransform() : glm::mat4(1.0f)) * local->GetLocalTransform()
            );
        }

        // Process each child node recursively
        for (auto c : p->GetChildNodes()) {
            _ProcessNode(c);
        }
    }
}