
#ifndef STRATUSGFX_ENTITY_H
#define STRATUSGFX_ENTITY_H

#include "StratusCommon.h"
#include "StratusHandle.h"
#include "StratusMath.h"
#include <any>
#include <memory>
#include <vector>

namespace stratus {
    #define BITMASK_OFFSET_POW2(offset) 1 << offset

    struct Entity;
    struct EntityComponent;

    typedef Handle<EntityComponent> EntityComponentHandle;
    typedef Handle<Entity> EntityHandle;
    typedef std::shared_ptr<EntityComponent> EntityComponentPtr;
    typedef std::shared_ptr<Entity> EntityPtr;

    enum class EntityEventType : uint64_t {
        // Generated for both the parent and the child.
        // The data will be std::shared_ptr<Entity> where for the child it will contain
        // the parent and for the parent it will contain the child.
        ENTITY_ATTACHED             = BITMASK_OFFSET_POW2(0),
        ENTITY_DETACHED             = BITMASK_OFFSET_POW2(1),
        // Next 3 involve a component of the entity's transform matrix changing. In this case
        // only Event.type will be valid, where data and deltaSeconds will be left as default.
        ENTITY_SCALE_CHANGED        = BITMASK_OFFSET_POW2(2),
        ENTITY_TRANSLATE_CHANGED    = BITMASK_OFFSET_POW2(3),
        ENTITY_ROTATION_CHANGED     = BITMASK_OFFSET_POW2(4),
        // Entity was added or removed from a currently-active Scene. Event.data will be equal to
        // shared_ptr<Scene> containins the Scene that the entity became active or inactive in.
        ENTITY_SPAWNED              = BITMASK_OFFSET_POW2(5),
        ENTITY_DESPAWNED            = BITMASK_OFFSET_POW2(6),
        // Following two happen when an entity is either deactivated (no longer receives updates)
        // or is reactivated (can receive updates)
        ENTITY_DEACTIVATED          = BITMASK_OFFSET_POW2(7),
        ENTITY_REACTIVATED          = BITMASK_OFFSET_POW2(8),
        // Called once per frame for each entity that has one or more components attached that
        // want to update. Event.deltaSeconds will be the time since the last frame.
        ENTITY_UPDATE               = BITMASK_OFFSET_POW2(9),
    };

    class Entity;
    // Contains some information to allow entity components to
    // process the event
    struct EntityEvent {
        EntityEvent(const std::shared_ptr<Entity>& entity, const EntityEventType& type)
            : entity(entity), type(type) {}

        EntityEvent(const std::shared_ptr<Entity>& entity, const EntityEventType& type, const double deltaSeconds)
            : entity(entity), type(type), deltaSeconds(deltaSeconds) {}

        EntityEvent(const std::shared_ptr<Entity>& entity, const EntityEventType& type, const std::any& data, const double deltaSeconds)
            : entity(entity), type(type), data(data), deltaSeconds(deltaSeconds) {}
        
        std::shared_ptr<Entity> entity;
        EntityEventType type;
        std::any data;
        double deltaSeconds;
    };

    // By itself an Entity is largely static and does not do much within a game world. To change
    // that, EntityComponents are added in order to embed custom game logic.
    struct EntityComponent {
        EntityComponent() : _handle(EntityComponentHandle::NextHandle()) {}
        virtual ~EntityComponent() = default;

        // Allows entity component to signal interest in events (set these before attaching, e.g. in constructor)
        void WatchEvent(const EntityEventType & type) { _eventMask |= static_cast<uint64_t>(type); }
        void UnwatchEvent(const EntityEventType & type) { _eventMask &= ~static_cast<uint64_t>(type); }
        uint64_t GetEventMask() const { return _eventMask; }

        // Called once per each event generated during the frame
        virtual void ProcessEvent(const EntityEvent& event) = 0;

        EntityComponentHandle GetHandle() const { return _handle; }

        // Deep copy of this component
        EntityComponentPtr Copy() const;
    
    private:
        // Specifies which events this component is interested in
        uint64_t _eventMask = 0;
        // Unique identifier for the component
        EntityComponentHandle _handle;
    };

    /**
     * Entity which stores relevant gameplay logic, rendering, physics and audio data. Only
     * one thread should handle an Entity tree at a time.
     */
    struct Entity {
        Entity();
        virtual ~Entity();

        // Functions for manipulating entity components
        virtual void AddComponent(const EntityComponentPtr&);
        virtual void RemoveComponent(const EntityComponentPtr&);
        virtual void RemoveComponentByHandle(const EntityComponentHandle&);
        virtual void RemoveAllComponents();
        virtual const std::vector<EntityComponentPtr>& GetComponents() const;

        // Functions for getting and setting transform
        virtual const glm::vec3& GetPosition() const;
        virtual void SetPosition(const glm::vec3&);

        virtual const Rotation& GetRotation() const;
        virtual void SetRotation(const Rotation&);

        virtual const glm::vec3& GetScale() const;
        virtual void SetScale(const glm::vec3&);

        // World transform is relative to the entity's parent
        virtual const glm::mat4& GetWorldTransform() const;
        // Local transform is only taking into account this current entity
        virtual const glm::mat4& GetLocalTransform() const;

        // Shows which events this entity needs to receive
        virtual uint64_t GetEventMask() const;
        // Called by the scene for each event that occurs
        virtual void SendEvent(const EntityEvent&);
        // Queueing and getting queued events
        virtual void QueueEvent(const EntityEvent&);
        virtual void GetQueuedEvents(std::vector<EntityEvent>& out);
        // Clears all events (happens after all events have been processed)
        virtual void ClearQueuedEvents();

        // Functions for dealing with parent and child entities
        virtual void SetParent(EntityPtr);
        virtual EntityPtr GetParent() const;

        virtual void AttachChild(EntityPtr);
        virtual void DetachChild(EntityPtr);
        virtual const std::vector<EntityPtr>& GetChildren() const;

        virtual EntityHandle GetHandle() const;

        // Functions for setting (optional) entity name and getting it
        virtual void SetName(const std::string&);
        virtual const std::string& GetName() const;

        // Creates a deep copy of this entity, its components, and all other nodes
        EntityPtr Copy() const;
    };
}

#endif //STRATUSGFX_ENTITY_H
