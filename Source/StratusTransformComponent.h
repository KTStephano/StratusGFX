#pragma once

#include "StratusEntity2.h"
#include "StratusEntityCommon.h"
#include "StratusEntityProcess.h"
#include "StratusUtils.h"
#include "StratusMath.h"

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace stratus {
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

        void SetLocalTransform(const glm::vec3& scale, const Rotation& rot, const glm::vec3& position);
        void SetLocalTransform(const glm::vec3& scale, const glm::mat3& rot, const glm::vec3& position);

        const glm::mat4& GetLocalTransform() const;

    private:
        void _MarkChangedAndRecalculate();

    private:
        glm::vec3 _scale = glm::vec3(1.0f);
        glm::mat3 _rotation = glm::mat3(1.0f);
        glm::vec3 _position = glm::vec3(0.0f);
        glm::mat4 _transform = glm::mat4(1.0f);
    };

    /*
    ENTITY_COMPONENT_STRUCT(GlobalTransformComponent)
        friend class TransformProcess;

        const glm::mat4& GetGlobalTransform() const;

    private:
        void _SetGlobalTransform(const glm::mat4&);

    private:
        glm::mat4 _transform = glm::mat4(1.0f);
    };
    */

    /*
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
        // For quick access without needing to query the entity
        std::unordered_map<Entity2Ptr, std::vector<Entity2Component *>> _components;
    };
    */
}