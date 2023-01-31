#include "StratusTransformComponent.h"

/*
    // Convenience functions for creating a new entity or initializing existing entity
    // with transform components
    extern Entity2Ptr CreateTransformEntity();
    extern void InitializeTransformEntity(Entity2Ptr);

    // Contains scale, rotate, translate for local coordinate system
    ENTITY_COMPONENT_STRUCT(LocalTransformComponent)
        void SetLocalScale(const glm::vec3&);
        const glm::vec3& GetLocalScale() const;

        void SetLocalRotation(const Rotation&);
        void SetLocalRotation(const glm::mat3&);
        const glm::mat3& GetLocalRotation() const;

        void SetLocalPosition(const glm::vec3&);
        const glm::vec3& GetLocalPosition() const;

        void SetLocalTransform(const glm::vec& scale, const Rotation& rot, const glm::vec3& position);
        void SetLocalTransform(const glm::vec& scale, const glm::mat3& rot, const glm::vec3& position);

        const glm::mat4& GetLocalTransform() const;

    private:
        glm::vec3 _scale = glm::vec3(1.0f);
        glm::mat3 _rotation = glm::mat3(1.0f);
        glm::vec3 _position = glm::vec3(0.0f);
        glm::mat4 _transform = glm::mat4(1.0f);
    };

    ENTITY_COMPONENT_STRUCT(GlobalTransformComponent)
        friend class TransformProcess;

        const glm::mat4& GetGlobalTransform() const;

    private:
        void _SetGlobalTransform(const glm::mat4&);

    private:
        glm::mat4 _transform = glm::mat4(1.0f);
    };

    class TransformProcess : public EntityProcess {
        virtual ~TransformProcess();

        void Process(const double deltaSeconds) override;
        void EntitiesAdded(const std::unordered_set<stratus::Entity2Ptr>&) override;
        void EntitiesRemoved(const std::unordered_set<stratus::Entity2Ptr>&) override;
        void EntityComponentsAdded(const std::unordered_map<stratus::Entity2Ptr, std::vector<stratus::Entity2Component *>>&) override;
        void EntityComponentsEnabledDisabled(const std::unordered_set<stratus::Entity2Ptr>&) override;

    private:
        static bool _IsEntityRelevant(const Entity2Ptr&);

    private:
        std::unordered_set<Entity2Ptr> _rootNodes;
        // For quick access without needing to query the component
        std::unordered_map<Entity2Ptr, std::vector<Entity2Component>> _components;
    */

namespace stratus {
    Entity2Ptr CreateTransformEntity() {
        auto ptr = Entity2::Create();
        InitializeTransformEntity(ptr);
        return ptr;
    }

    void InitializeTransformEntity(Entity2Ptr ptr) {
        ptr->Components().AttachComponent<LocalTransformComponent>();
        //ptr->Components().AttachComponent<GlobalTransformComponent>();
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
}