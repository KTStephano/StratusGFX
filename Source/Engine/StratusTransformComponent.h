#pragma once

#include "StratusEntity.h"
#include "StratusEntityCommon.h"
#include "StratusEntityProcess.h"
#include "StratusUtils.h"
#include "StratusMath.h"

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <mutex>

namespace stratus {
    // Convenience functions for creating a new entity or initializing existing entity
    // with transform components
    extern EntityPtr CreateTransformEntity();
    extern void InitializeTransformEntity(EntityPtr);

    // Contains scale, rotate, translate for local coordinate system
    ENTITY_COMPONENT_STRUCT(LocalTransformComponent)
        LocalTransformComponent() = default;
        LocalTransformComponent(const LocalTransformComponent&) = default;

        void SetLocalScale(const glm::vec3&);
        const glm::vec3& GetLocalScale() const;

        void SetLocalRotation(const Rotation&);
        void SetLocalRotation(const glm::mat3&);
        const glm::mat3& GetLocalRotation() const;

        void SetLocalPosition(const glm::vec3&);
        const glm::vec3& GetLocalPosition() const;

        void SetLocalTransform(const glm::vec3& scale, const Rotation& rot, const glm::vec3& position);
        void SetLocalTransform(const glm::vec3& scale, const glm::mat3& rot, const glm::vec3& position);

        const glm::mat4& GetLocalTransform() const;

    private:
        void MarkChangedAndRecalculate_();

    private:
        glm::vec3 scale_ = glm::vec3(1.0f);
        glm::mat3 rotation_ = glm::mat3(1.0f);
        glm::vec3 position_ = glm::vec3(0.0f);
        glm::mat4 transform_ = glm::mat4(1.0f);
    };

    ENTITY_COMPONENT_STRUCT(GlobalTransformComponent)
        friend class TransformProcess;

        GlobalTransformComponent() = default;
        GlobalTransformComponent(const GlobalTransformComponent&) = default;

        const glm::mat4& GetGlobalTransform() const;

    private:
        void SetGlobalTransform_(const glm::mat4&);

    private:
        glm::mat4 transform_ = glm::mat4(1.0f);
    };

    class TransformProcess : public EntityProcess {
        virtual ~TransformProcess();

        void Process(const double deltaSeconds) override;
        void EntitiesAdded(const std::unordered_set<stratus::EntityPtr>&) override;
        void EntitiesRemoved(const std::unordered_set<stratus::EntityPtr>&) override;
        void EntityComponentsAdded(const std::unordered_map<stratus::EntityPtr, std::vector<stratus::EntityComponent *>>&) override;
        void EntityComponentsEnabledDisabled(const std::unordered_set<stratus::EntityPtr>&) override;

    private:
        static bool IsEntityRelevant_(const EntityPtr&);

    private:
        void ProcessNode_(const EntityPtr& p);

    private:
        std::unordered_set<EntityPtr> rootNodes_;
        // For quick access without needing to query the entity
        std::unordered_map<EntityPtr, std::vector<EntityComponent *>> components_;
    };
}