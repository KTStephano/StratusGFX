
#ifndef STRATUSGFX_ENTITY_H
#define STRATUSGFX_ENTITY_H

#include "StratusCommon.h"
#include "StratusHandle.h"
#include "StratusMath.h"
#include "StratusRenderNode.h"
#include <any>
#include <memory>
#include <vector>
#include <unordered_set>
#include <atomic>

namespace stratus {
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
        ENTITY_ATTACHED             = BITMASK64_POW2(0),
        ENTITY_DETACHED             = BITMASK64_POW2(1),
        // Next 3 involve a component of the entity's transform matrix changing. In this case
        // only Event.type will be valid, where data and deltaSeconds will be left as default.
        ENTITY_SCALE_CHANGED        = BITMASK64_POW2(2),
        ENTITY_TRANSLATE_CHANGED    = BITMASK64_POW2(3),
        ENTITY_ROTATION_CHANGED     = BITMASK64_POW2(4),
        // Entity was added or removed from a currently-active Scene. Event.data will be equal to
        // shared_ptr<Scene> containins the Scene that the entity became active or inactive in.
        ENTITY_SPAWNED              = BITMASK64_POW2(5),
        ENTITY_DESPAWNED            = BITMASK64_POW2(6),
        // Following two happen when an entity is either deactivated (no longer receives updates)
        // or is reactivated (can receive updates)
        ENTITY_DEACTIVATED          = BITMASK64_POW2(7),
        ENTITY_REACTIVATED          = BITMASK64_POW2(8),
        // Called once per frame for each entity that has one or more components attached that
        // want to update. Event.deltaSeconds will be the time since the last frame.
        ENTITY_UPDATE               = BITMASK64_POW2(9),
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
        virtual EntityComponentPtr Copy() const = 0;
    
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
    struct Entity : public std::enable_shared_from_this<Entity> {
        Entity();

    public:
        static EntityPtr Create();

        virtual ~Entity();

        // Functions for manipulating entity components
        virtual void AddComponent(const EntityComponentPtr&);
        virtual void RemoveComponent(const EntityComponentPtr&);
        virtual void RemoveComponentByHandle(const EntityComponentHandle&);
        virtual void RemoveAllComponents();
        virtual const std::vector<EntityComponentPtr>& GetComponents() const;

        // Functions for getting and setting transform
        virtual const glm::vec3& GetLocalPosition() const;
        virtual void SetLocalPosition(const glm::vec3&);

        virtual const Rotation& GetLocalRotation() const;
        virtual void SetLocalRotation(const Rotation&);

        virtual const glm::vec3& GetLocalScale() const;
        virtual void SetLocalScale(const glm::vec3&);

        virtual void SetLocalPosRotScale(const glm::vec3&, const Rotation&, const glm::vec3&);

        virtual const glm::vec3& GetWorldPosition() const;

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
        virtual EntityPtr GetParent() const;

        virtual void AttachChild(EntityPtr);
        virtual void DetachChild(EntityPtr);
        virtual const std::unordered_set<EntityPtr>& GetChildren() const;

        EntityHandle GetHandle() const;
        uint64_t GetRefCount() const;

        // Functions for setting (optional) entity name and getting it
        virtual void SetName(const std::string&);
        virtual const std::string& GetName() const;

        RenderNodePtr GetRenderNode() const;
        void SetRenderNode(const RenderNodePtr&);

        // Creates a deep copy of this entity, its components, and all other nodes
        // (if copyHandle == false it will generated a new handle)
        virtual EntityPtr Copy(bool copyHandle = false) const;

        virtual bool operator==(const Entity& other) const {
            return _handle == other._handle;
        }

        virtual size_t HashCode() const {
            return std::hash<EntityHandle>{}(_handle);
        }

    protected:
        virtual void Copy(EntityPtr&, bool) const;
        void IncrRefCount();
        void DecrRefCount();

    private:
        void _SetParent(EntityPtr);
        void _RecalcTransform();

    private:
        EntityPtr _parent;
        EntityHandle _handle;
        std::shared_ptr<std::atomic<uint64_t>> _refCount;
        std::unordered_set<EntityPtr> _children;
        RenderNodePtr _renderNode;
        glm::vec3 _position = glm::vec3(0.0f);
        glm::vec3 _worldPosition;
        Rotation _rotation;
        glm::vec3 _scale = glm::vec3(1.0f);
        glm::mat4 _localTransform = glm::mat4(1.0f);
        glm::mat4 _worldScale = glm::mat4(1.0f);
        glm::mat4 _worldRotate = glm::mat4(1.0f);
        glm::mat4 _worldTranslate = glm::mat4(1.0f);
        glm::mat4 _worldTransform = glm::mat4(1.0f);
    };

    // Allows for entity to be inserted into hash map
    struct EntityView {
        EntityView() {}
        EntityView(const EntityPtr& entity) : _entity(entity) {}
        const EntityPtr& Get() const { return _entity; }

        bool operator==(const EntityView& other) const {
            return *_entity == *other._entity;
        }

        bool operator!=(const EntityView& other) const {
            return !(*this == other);
        }

        size_t HashCode() const {
            return _entity->HashCode();
        }

    private:
        EntityPtr _entity;
    };
}

namespace std {
    template<>
    struct hash<stratus::EntityView> {
        size_t operator()(const stratus::EntityView& v) const {
            return v.HashCode();
        }
    };
}

#endif //STRATUSGFX_ENTITY_H
