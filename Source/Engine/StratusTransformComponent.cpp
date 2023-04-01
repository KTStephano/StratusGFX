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
        SetLocalTransform(scale, rotation_, position_);
    }

    const glm::vec3& LocalTransformComponent::GetLocalScale() const {
        return scale_;
    }

    void LocalTransformComponent::SetLocalRotation(const Rotation& rot) {
        SetLocalRotation(rot.asMat3());
    }

    void LocalTransformComponent::SetLocalRotation(const glm::mat3& m) {
        SetLocalTransform(scale_, m, position_);
    }

    const glm::mat3& LocalTransformComponent::GetLocalRotation() const {
        return rotation_;
    }

    void LocalTransformComponent::SetLocalPosition(const glm::vec3& position) {
        SetLocalTransform(scale_, rotation_, position);
    }

    const glm::vec3& LocalTransformComponent::GetLocalPosition() const {
        return position_;
    }

    void LocalTransformComponent::SetLocalTransform(const glm::vec3& scale, const Rotation& rot, const glm::vec3& position) {
        SetLocalTransform(scale, rot.asMat3(), position);
    }

    void LocalTransformComponent::SetLocalTransform(const glm::vec3& scale, const glm::mat3& rot, const glm::vec3& position) {
        if (&scale != &scale_) scale_ = scale;
        if (&rot != &rotation_) rotation_ = rot;
        if (&position != &position_) position_ = position;
        MarkChangedAndRecalculate_();
    }

    const glm::mat4& LocalTransformComponent::GetLocalTransform() const {
        return transform_;
    }

    void LocalTransformComponent::MarkChangedAndRecalculate_() {
        this->MarkChanged();
        auto S = glm::mat4(1.0f);
        matScale(S, scale_);
        auto R = glm::mat4(1.0f);
        matInset(R, rotation_); 
        auto T = glm::mat4(1.0f);
        matTranslate(T, position_);
        transform_ = T * R * S;
        /*
        glm::mat3 RS = _rotation;
        matScale(RS, _scale);
        _transform = glm::mat4(1.0f);
        matTranslate(_transform, _position);
        matInset(_transform, RS);
        */
    }

    const glm::mat4& GlobalTransformComponent::GetGlobalTransform() const {
        return transform_;
    }

    void GlobalTransformComponent::SetGlobalTransform_(const glm::mat4& m) {
        this->MarkChanged();
        transform_ = m;
    }

    TransformProcess::~TransformProcess() {}

    void TransformProcess::Process(const double deltaSeconds) {
        for (auto root : rootNodes_) {
            ProcessNode_(root);
        }
    }

    void TransformProcess::EntitiesAdded(const std::unordered_set<stratus::EntityPtr>& entities) {
        for (auto ptr : entities) {
            if (IsEntityRelevant_(ptr)) {
                std::vector<EntityComponent *> components{
                    ptr->Components().GetComponent<LocalTransformComponent>().component,
                    ptr->Components().GetComponent<GlobalTransformComponent>().component
                };
                
                components_.insert(std::make_pair(ptr, std::move(components)));

                if (ptr->GetParentNode() == nullptr) {
                    rootNodes_.insert(ptr);
                    ProcessNode_(ptr);
                }
            }
        }
    }

    void TransformProcess::EntitiesRemoved(const std::unordered_set<stratus::EntityPtr>& entities) {
        for (auto ptr : entities) {
            rootNodes_.erase(ptr);
            components_.erase(ptr);
        }
    }

    void TransformProcess::EntityComponentsAdded(const std::unordered_map<stratus::EntityPtr, std::vector<stratus::EntityComponent *>>& entities) {
        std::unordered_set<EntityPtr> added;
        for (auto p : entities) {
            if (IsEntityRelevant_(p.first)) {
                added.insert(p.first);
            }
        }

        EntitiesAdded(added);
    }

    void TransformProcess::EntityComponentsEnabledDisabled(const std::unordered_set<stratus::EntityPtr>& entities) {
        std::unordered_set<EntityPtr> added;
        std::unordered_set<EntityPtr> removed;
        for (auto ptr : entities) {
            if (IsEntityRelevant_(ptr)) {
                added.insert(ptr);
            }
            else {
                removed.insert(ptr);
            }
        }

        EntitiesAdded(added);
        EntitiesRemoved(removed);
    }

    bool TransformProcess::IsEntityRelevant_(const EntityPtr& e) {
        auto& components = e->Components();
        auto local = components.GetComponent<LocalTransformComponent>();
        auto global = components.GetComponent<GlobalTransformComponent>();
        if (local.component == nullptr || global.component == nullptr) return false;
        return local.status == EntityComponentStatus::COMPONENT_ENABLED &&
            global.status == EntityComponentStatus::COMPONENT_ENABLED;
    }

    void TransformProcess::ProcessNode_(const EntityPtr& p) {
        auto it = components_.find(p);
        if (it == components_.end()) return;

        bool parentChanged = false;
        auto parent = p->GetParentNode();
        GlobalTransformComponent * parentGlobal = nullptr;
        if (parent != nullptr) {
            auto itparent = components_.find(parent);
            if (itparent != components_.end()) {
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
            global->SetGlobalTransform_(
                (parentGlobal ? parentGlobal->GetGlobalTransform() : glm::mat4(1.0f)) * local->GetLocalTransform()
            );
        }

        // Process each child node recursively
        for (auto c : p->GetChildNodes()) {
            ProcessNode_(c);
        }
    }
}