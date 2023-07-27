#pragma once

#include "StratusHandle.h"
#include <unordered_set>
#include <unordered_map>
#include <typeinfo>
#include <vector>
#include <string>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include "StratusEntityCommon.h"
#include "StratusPoolAllocator.h"

template<typename E>
std::string ClassName() {
    return typeid(E).name();
}

template<typename E>
size_t ClassHashCode() {
    return typeid(E).hash_code();
}

template<typename Component>
struct ComponentAllocator_ {
    std::mutex m;
    stratus::PoolAllocator<Component> allocator;
};

template<typename Component>
ComponentAllocator_<Component>& GetComponentAllocator_() {
    static ComponentAllocator_<Component> allocator;
    return allocator;
}

#define ENTITY_COMPONENT_STRUCT(name)                                                       \
    struct name final : public stratus::EntityComponent {                                   \
        static std::string STypeName() { return ClassName<name>(); }                        \
        static size_t SHashCode() { return ClassHashCode<name>(); }                         \
        std::string TypeName() const override { return STypeName(); }                       \
        size_t HashCode() const override { return SHashCode(); }                            \
        bool operator==(const stratus::EntityComponent * other) const override {            \
            if (this == other) return true;                                                 \
            if (!other) return false;                                                       \
            return TypeName() == std::string(other->TypeName());                            \
        }                                                                                   \
        stratus::EntityComponent * Copy() const override {                                  \
            auto ptr = dynamic_cast<stratus::EntityComponent *>(name::Create(*this));       \
            return ptr;                                                                     \
        }                                                                                   \
        template<typename ... Types>                                                        \
        static name * Create(const Types& ... args) {                                       \
            auto& allocator = GetComponentAllocator_<name>();                               \
            auto ul = std::unique_lock(allocator.m);                                        \
            return allocator.allocator.AllocateConstruct(args...);                                   \
        }                                                                                   \
        static void Destroy(name * ptr) {                                                   \
            if (ptr == nullptr) return;                                                     \
            auto& allocator = GetComponentAllocator_<name>();                               \
            auto ul = std::unique_lock(allocator.m);                                        \
            allocator.allocator.DestroyDeallocate(ptr);                                            \
        }

namespace stratus {
    class Entity;
    typedef Handle<Entity> EntityHandle;

    enum class EntityComponentStatus : int64_t {
        COMPONENT_ENABLED,
        COMPONENT_DISABLED
    };

    // Meant to store any relevant data that will be manipulated by the engine + application
    // at runtime
    struct EntityComponent {
        virtual ~EntityComponent() = default;
        virtual std::string TypeName() const = 0;
        virtual size_t HashCode() const = 0;
        virtual bool operator==(const EntityComponent *) const = 0;

        virtual EntityComponent * Copy() const = 0;

        void MarkChanged();
        bool ChangedLastFrame() const;
        bool ChangedThisFrame() const;
        bool ChangedWithinLastFrame() const;

    protected:
        // Last engine frame this component was modified
        uint64_t lastFrameChanged_ = 0;

    };

    // Enables an entity component pointer to be inserted into hash set/map
    struct EntityComponentView {
        EntityComponent * component = nullptr;

        EntityComponentView(EntityComponent * c = nullptr)
            : component(c) {}

        size_t HashCode() const {
            if (!component) return 0;
            return component->HashCode();
        }

        bool operator==(const EntityComponentView& other) const {
            if (component == other.component) return true;
            if (!component) return false;

            return other.component->operator==(component);
        }
    };

    struct EntityComponentPointerManager {
        EntityComponent * component = nullptr;

        EntityComponentPointerManager(EntityComponent * c)
            : component(c) {}

        virtual ~EntityComponentPointerManager() = default;

        virtual EntityComponentPointerManager * Copy() const = 0;
    };

    template<typename Component, typename ... Types>
    std::unique_ptr<EntityComponentPointerManager> ConstructComponent_(const Types& ... args) {
        struct Pointer_ final : public EntityComponentPointerManager {
            Pointer_(EntityComponent * c)
                : EntityComponentPointerManager(c) {}

            virtual ~Pointer_() {
                Component::Destroy(dynamic_cast<Component *>(component));
            }

            EntityComponentPointerManager * Copy() const override {
                return new Pointer_(component->Copy());
            }
        };

        return std::unique_ptr<EntityComponentPointerManager>(new Pointer_(Component::Create(args...)));
    }

    inline std::unique_ptr<EntityComponentPointerManager> CopyManager_(const std::unique_ptr<EntityComponentPointerManager>& ptr) {
        return std::unique_ptr<EntityComponentPointerManager>(ptr->Copy());
    }
}

namespace std {
    template<>
    struct hash<stratus::EntityComponentView> {
        size_t operator()(const stratus::EntityComponentView & v) const {
            return v.HashCode();
        }
    };
}

namespace stratus {
    template<typename E>
    struct EntityComponentPair {
        E * component = nullptr;
        EntityComponentStatus status = EntityComponentStatus::COMPONENT_DISABLED;
    };

    // Keeps track of a unique set of components
    // Not thread safe unless doing readonly operations
    // Guarantee: Component pointers will never move around in memory even when new ones are added
    struct EntityComponentSet final {
        friend class Entity;

        ~EntityComponentSet();

        // Manipulate components - thread safe
        template<typename E, typename ... Types>
        void AttachComponent(const Types& ... args);

        template<typename E>
        bool ContainsComponent() const;

        template<typename E>
        EntityComponentPair<E> GetComponent();

        template<typename E>
        EntityComponentPair<const E> GetComponent() const;

        EntityComponentPair<EntityComponent> GetComponentByName(const std::string&);
        EntityComponentPair<const EntityComponent> GetComponentByName(const std::string&) const;

        template<typename E>
        void EnableComponent();
        
        template<typename E>
        void DisableComponent();

        std::vector<EntityComponentPair<EntityComponent>> GetAllComponents();
        std::vector<EntityComponentPair<const EntityComponent>> GetAllComponents() const;

        static EntityComponentSet * Create() {
            return new EntityComponentSet();
        }

        static void Destroy(EntityComponentSet * ptr) {
            delete ptr;
        }

        EntityComponentSet * Copy() const;

    private:
        template<typename E>
        EntityComponentPair<E> GetComponent_() const;

        template<typename E>
        EntityComponentPair<E> GetComponentByName_(const std::string&) const;

        template<typename E, typename ... Types>
        void AttachComponent_(const Types& ... args);

        template<typename E>
        bool ContainsComponent_() const;

        template<typename E>
        void SetComponentStatus_(EntityComponentStatus);

        void AttachComponent_(std::unique_ptr<EntityComponentPointerManager>&);
        void SetOwner_(Entity *);
        void NotifyEntityManagerComponentEnabledDisabled_();

    private:
        //mutable std::shared_mutex _m;
        Entity * owner_;
        // Component pointer managers (allocates and deallocates from shared pool)
        std::vector<std::unique_ptr<EntityComponentPointerManager>> componentManagers_;
        // List of unique components
        std::unordered_set<EntityComponentView> components_;
        // List of components by type name
        std::unordered_map<std::string, std::pair<EntityComponentView, EntityComponentStatus>> componentTypeNames_;
    };

    // Collection of unque ID + configurable component data
    class Entity final : public std::enable_shared_from_this<Entity> {
        friend class EntityManager;

        Entity();
        Entity(EntityComponentSet *);

        // Since our constructor is private we provide the pool allocator with this
        // function to bypass it
        template<typename ... Types>
        static Entity * PlacementNew_(uint8_t * memory, const Types& ... args) {
            return new (memory) Entity(args...);
        }

    public:
        static EntityPtr Create();

    private:
        static EntityPtr Create(EntityComponentSet *);

    public:
        ~Entity();

        Entity(Entity&&) = delete;
        Entity(const Entity&) = delete;
        Entity& operator=(Entity&&) = delete;
        Entity& operator=(const Entity&) = delete;

        EntityPtr Copy() const;

        // Manipulate components - thread safe
        EntityComponentSet& Components();
        const EntityComponentSet& Components() const;

        bool IsInWorld() const;

        // This is supported when the entity is not part of the world
        // Once added its tree structure becomes immutable
        //
        // Attaching/Detaching nodes is NOT thread safe
        void AttachChildNode(const EntityPtr&);
        void DetachChildNode(const EntityPtr&);
        EntityPtr GetParentNode() const;
        const std::vector<EntityPtr>& GetChildNodes() const;
        bool ContainsChildNode(const EntityPtr&) const;

        const EntityHandle& GetHandle() const;

    private:
        // Called by EntityManager class
        void AddToWorld_();
        void RemoveFromWorld_();

    private:
        bool ContainsChildNode_(const EntityPtr&) const;

    private:
        mutable std::shared_mutex m_;
        EntityHandle handle_;
        // List of unique components
        EntityComponentSet * components_;
        EntityWeakPtr parent_;
        std::vector<EntityPtr> childNodes_;
        // Keeps track of added/removed from world
        bool partOfWorld_ = false;
    };

    // Convenience functions
    template<typename E>
    bool ContainsComponent(const EntityPtr& p) {
        return p->Components().ContainsComponent<E>();
    }

    template<typename E>
    E * GetComponent(const EntityPtr& p) {
        return p->Components().GetComponent<E>().component;
    }

    template<typename E>
    EntityComponentStatus GetComponentStatus(const EntityPtr& p) {
        return p->Components().GetComponent<E>().status;
    }

    template<typename E>
    EntityComponentPair<E> GetComponentStatusPair(const EntityPtr& p) {
        return p->Components().GetComponent<E>();
    }

    template<typename E, typename ... Types>
    void EntityComponentSet::AttachComponent(const Types& ... args) {
        static_assert(std::is_base_of<EntityComponent, E>::value);
        //auto ul = std::unique_lock<std::shared_mutex>(_m);
        return AttachComponent_<E>(args...);
    }

    template<typename E>
    bool EntityComponentSet::ContainsComponent() const {
        static_assert(std::is_base_of<EntityComponent, E>::value);
        //auto sl = std::shared_lock<std::shared_mutex>(_m);
        return ContainsComponent_<E>();
    }

    template<typename E, typename ... Types>
    void EntityComponentSet::AttachComponent_(const Types& ... args) {
        if (ContainsComponent_<E>()) return;
        auto ptr = ConstructComponent_<E>(args...);
        AttachComponent_(ptr);
    }

    template<typename E>
    bool EntityComponentSet::ContainsComponent_() const {
        std::string name = E::STypeName();
        return componentTypeNames_.find(name) != componentTypeNames_.end();
    }

    template<typename E>
    EntityComponentPair<E> EntityComponentSet::GetComponent() {
        return GetComponent_<E>();
    }

    template<typename E>
    EntityComponentPair<const E> EntityComponentSet::GetComponent() const {
        return GetComponent_<const E>();
    }

    template<typename E>
    EntityComponentPair<E> EntityComponentSet::GetComponent_() const {
        static_assert(std::is_base_of<EntityComponent, E>::value);
        std::string name = E::STypeName();
        return GetComponentByName_<E>(name);
    }

    template<typename E>
    EntityComponentPair<E> EntityComponentSet::GetComponentByName_(const std::string& name) const {
        static_assert(std::is_base_of<EntityComponent, E>::value);
        //auto sl = std::shared_lock<std::shared_mutex>(_m);
        auto it = componentTypeNames_.find(name);
        return it != componentTypeNames_.end() ? 
            EntityComponentPair<E>{dynamic_cast<E *>(it->second.first.component), it->second.second} : 
            EntityComponentPair<E>();
    }

    template<typename E>
    void EntityComponentSet::EnableComponent() {
        SetComponentStatus_<E>(EntityComponentStatus::COMPONENT_ENABLED);
    }
    
    template<typename E>
    void EntityComponentSet::DisableComponent() {
        SetComponentStatus_<E>(EntityComponentStatus::COMPONENT_DISABLED);
    }

    template<typename E>
    void EntityComponentSet::SetComponentStatus_(EntityComponentStatus status) {
        static_assert(std::is_base_of<EntityComponent, E>::value);
        //auto ul = std::unique_lock<std::shared_mutex>(_m);
        std::string name = E::STypeName();
        auto it = componentTypeNames_.find(name);
        if (it != componentTypeNames_.end()) {
            if (it->second.second != status) {
                it->second.second = status;
                NotifyEntityManagerComponentEnabledDisabled_();
            }
        }
    }
}