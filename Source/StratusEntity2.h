#pragma once

#include "StratusHandle.h"
#include <unordered_set>
#include <unordered_map>
#include <typeinfo>
#include <vector>
#include <string>
#include <memory>
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
struct __ComponentAllocator {
    std::mutex m;
    stratus::PoolAllocator<Component> allocator;
};

template<typename Component>
__ComponentAllocator<Component>& __GetComponentAllocator() {
    static __ComponentAllocator<Component> allocator;
    return allocator;
}

#define ENTITY_COMPONENT_STRUCT(name)                                                       \
    struct name final : public stratus::Entity2Component {                                  \
        static std::string STypeName() { return ClassName<name>(); }                        \
        static size_t SHashCode() { return ClassHashCode<name>(); }                         \
        std::string TypeName() const override { return STypeName(); }                       \
        size_t HashCode() const override { return SHashCode(); }                            \
        bool operator==(const stratus::Entity2Component * other) const override {           \
            if (*this == other) return true;                                                \
            if (!other) return false;                                                       \
            return TypeName() == std::string(other->TypeName());                            \
        }                                                                                   \
        stratus::Entity2Component * Copy() const override {                                 \
            auto ptr = dynamic_cast<stratus::Entity2Component *>(name::Create(*this));      \
            return ptr;                                                                     \
        }                                                                                   \
        template<typename ... Types>                                                        \
        static name * Create(const Types& ... args) {                                       \
            auto& allocator = __GetComponentAllocator<name>();                              \
            auto ul = std::unique_lock(allocator.m);                                        \
            return allocator.allocator.Allocate(args...);                                   \
        }                                                                                   \
        static void Destroy(name * ptr) {                                                   \
            if (ptr == nullptr) return;                                                     \
            auto& allocator = __GetComponentAllocator<name>();                              \
            auto ul = std::unique_lock(allocator.m);                                        \
            allocator.allocator.Deallocate(ptr);                                            \
        }

namespace stratus {
    enum class EntityComponentStatus : int64_t {
        COMPONENT_ENABLED,
        COMPONENT_DISABLED
    };

    // Meant to store any relevant data that will be manipulated by the engine + application
    // at runtime
    struct Entity2Component {
        virtual ~Entity2Component() = default;
        virtual std::string TypeName() const = 0;
        virtual size_t HashCode() const = 0;
        virtual bool operator==(const Entity2Component *) const = 0;

        virtual Entity2Component * Copy() const = 0;

        void MarkChanged();
        bool ChangedLastFrame() const;
        bool ChangedThisFrame() const;
        bool ChangedWithinLastFrame() const;

    protected:
        // Last engine frame this component was modified
        uint64_t _lastFrameChanged = 0;

    };

    // Enables an entity component pointer to be inserted into hash set/map
    struct Entity2ComponentView {
        Entity2Component * component = nullptr;

        Entity2ComponentView(Entity2Component * c = nullptr)
            : component(c) {}

        size_t HashCode() const {
            if (!component) return 0;
            return component->HashCode();
        }

        bool operator==(const Entity2ComponentView& other) const {
            if (component == other.component) return true;
            if (!component) return false;

            return other.component->operator==(component);
        }
    };

    struct EntityComponentPointerManager {
        Entity2Component * component = nullptr;

        EntityComponentPointerManager(Entity2Component * c)
            : component(c) {}

        virtual ~EntityComponentPointerManager() = default;

        virtual EntityComponentPointerManager * Copy() const = 0;
    };

    template<typename Component, typename ... Types>
    std::unique_ptr<EntityComponentPointerManager> __ConstructComponent(const Types& ... args) {
        struct _Pointer final : public EntityComponentPointerManager {
            _Pointer(Entity2Component * c)
                : EntityComponentPointerManager(c) {}

            virtual ~_Pointer() {
                Component::Destroy(dynamic_cast<Component *>(component));
            }

            EntityComponentPointerManager * Copy() const override {
                return new _Pointer(component->Copy());
            }
        };

        return std::unique_ptr<EntityComponentPointerManager>(new _Pointer(Component::Create(args...)));
    }

    inline std::unique_ptr<EntityComponentPointerManager> __CopyManager(const std::unique_ptr<EntityComponentPointerManager>& ptr) {
        return std::unique_ptr<EntityComponentPointerManager>(ptr->Copy());
    }
}

namespace std {
    template<>
    struct hash<stratus::Entity2ComponentView> {
        size_t operator()(const stratus::Entity2ComponentView & v) const {
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
    struct Entity2ComponentSet final {
        friend class Entity2;

        ~Entity2ComponentSet();

        // Manipulate components - thread safe
        template<typename E, typename ... Types>
        void AttachComponent(const Types& ... args);

        template<typename E>
        bool ContainsComponent() const;

        template<typename E>
        EntityComponentPair<E> GetComponent();

        template<typename E>
        EntityComponentPair<const E> GetComponent() const;

        EntityComponentPair<Entity2Component> GetComponentByName(const std::string&);
        EntityComponentPair<const Entity2Component> GetComponentByName(const std::string&) const;

        template<typename E>
        void EnableComponent();
        
        template<typename E>
        void DisableComponent();

        std::vector<EntityComponentPair<Entity2Component>> GetAllComponents();
        std::vector<EntityComponentPair<const Entity2Component>> GetAllComponents() const;

        static Entity2ComponentSet * Create() {
            return new Entity2ComponentSet();
        }

        static void Destroy(Entity2ComponentSet * ptr) {
            delete ptr;
        }

        Entity2ComponentSet * Copy() const;

    private:
        template<typename E>
        EntityComponentPair<E> _GetComponent() const;

        template<typename E>
        EntityComponentPair<E> _GetComponentByName(const std::string&) const;

        template<typename E, typename ... Types>
        void _AttachComponent(const Types& ... args);

        template<typename E>
        bool _ContainsComponent() const;

        template<typename E>
        void _SetComponentStatus(EntityComponentStatus);

        void _AttachComponent(std::unique_ptr<EntityComponentPointerManager>&);
        void _SetOwner(Entity2 *);
        void _NotifyEntityManagerComponentEnabledDisabled();

    private:
        //mutable std::shared_mutex _m;
        Entity2 * _owner;
        // Component pointer managers (allocates and deallocates from shared pool)
        std::vector<std::unique_ptr<EntityComponentPointerManager>> _componentManagers;
        // List of unique components
        std::unordered_set<Entity2ComponentView> _components;
        // List of components by type name
        std::unordered_map<std::string, std::pair<Entity2ComponentView, EntityComponentStatus>> _componentTypeNames;
    };

    // Collection of unque ID + configurable component data
    class Entity2 final : public std::enable_shared_from_this<Entity2> {
        friend class EntityManager;

        Entity2();
        Entity2(Entity2ComponentSet *);

        // Since our constructor is private we provide the pool allocator with this
        // function to bypass it
        template<typename ... Types>
        static Entity2 * _PlacementNew(uint8_t * memory, const Types& ... args) {
            return new (memory) Entity2(args...);
        }

    public:
        static Entity2Ptr Create();

    private:
        static Entity2Ptr Create(Entity2ComponentSet *);

    public:
        ~Entity2();

        Entity2Ptr Copy() const;

        // Manipulate components - thread safe
        Entity2ComponentSet& Components();
        const Entity2ComponentSet& Components() const;

        bool IsInWorld() const;

        // This is supported when the entity is not part of the world
        // Once added its tree structure becomes immutable
        //
        // Attaching/Detaching nodes is NOT thread safe
        void AttachChildNode(const Entity2Ptr&);
        void DetachChildNode(const Entity2Ptr&);
        Entity2Ptr GetParentNode() const;
        const std::vector<Entity2Ptr>& GetChildNodes() const;
        bool ContainsChildNode(const Entity2Ptr&) const;

    private:
        // Called by EntityManager class
        void _AddToWorld();
        void _RemoveFromWorld();

    private:
        bool _ContainsChildNode(const Entity2Ptr&) const;

    private:
        mutable std::shared_mutex _m;
        // List of unique components
        Entity2ComponentSet * _components;
        Entity2WeakPtr _parent;
        std::vector<Entity2Ptr> _childNodes;
        // Keeps track of added/removed from world
        bool _partOfWorld = false;
    };

    template<typename E, typename ... Types>
    void Entity2ComponentSet::AttachComponent(const Types& ... args) {
        static_assert(std::is_base_of<Entity2Component, E>::value);
        //auto ul = std::unique_lock<std::shared_mutex>(_m);
        return _AttachComponent<E>(args...);
    }

    template<typename E>
    bool Entity2ComponentSet::ContainsComponent() const {
        static_assert(std::is_base_of<Entity2Component, E>::value);
        //auto sl = std::shared_lock<std::shared_mutex>(_m);
        return _ContainsComponent<E>();
    }

    template<typename E, typename ... Types>
    void Entity2ComponentSet::_AttachComponent(const Types& ... args) {
        if (_ContainsComponent<E>()) return;
        auto ptr = __ConstructComponent<E>(args...);
        _AttachComponent(ptr);
    }

    template<typename E>
    bool Entity2ComponentSet::_ContainsComponent() const {
        std::string name = E::STypeName();
        return _componentTypeNames.find(name) != _componentTypeNames.end();
    }

    template<typename E>
    EntityComponentPair<E> Entity2ComponentSet::GetComponent() {
        return _GetComponent<E>();
    }

    template<typename E>
    EntityComponentPair<const E> Entity2ComponentSet::GetComponent() const {
        return _GetComponent<const E>();
    }

    template<typename E>
    EntityComponentPair<E> Entity2ComponentSet::_GetComponent() const {
        static_assert(std::is_base_of<Entity2Component, E>::value);
        std::string name = E::STypeName();
        return _GetComponentByName<E>(name);
    }

    template<typename E>
    EntityComponentPair<E> Entity2ComponentSet::_GetComponentByName(const std::string& name) const {
        static_assert(std::is_base_of<Entity2Component, E>::value);
        //auto sl = std::shared_lock<std::shared_mutex>(_m);
        auto it = _componentTypeNames.find(name);
        return it != _componentTypeNames.end() ? 
            EntityComponentPair<E>{dynamic_cast<E *>(it->second.first.component), it->second.second} : 
            EntityComponentPair<E>();
    }

    template<typename E>
    void Entity2ComponentSet::EnableComponent() {
        _SetComponentStatus<E>(EntityComponentStatus::COMPONENT_ENABLED);
    }
    
    template<typename E>
    void Entity2ComponentSet::DisableComponent() {
        _SetComponentStatus<E>(EntityComponentStatus::COMPONENT_DISABLED);
    }

    template<typename E>
    void Entity2ComponentSet::_SetComponentStatus(EntityComponentStatus status) {
        static_assert(std::is_base_of<Entity2Component, E>::value);
        //auto ul = std::unique_lock<std::shared_mutex>(_m);
        std::string name = E::STypeName();
        auto it = _componentTypeNames.find(name);
        if (it != _componentTypeNames.end()) {
            if (it->second.second != status) {
                it->second.second = status;
                _NotifyEntityManagerComponentEnabledDisabled();
            }
        }
    }
}