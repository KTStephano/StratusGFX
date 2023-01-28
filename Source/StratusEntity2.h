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

#define ENTITY_COMPONENT_STRUCT(name)                                                       \
    struct name : public stratus::Entity2Component {                                        \
        static std::string STypeName() { return ClassName<name>(); }                        \
        static size_t SHashCode() { return ClassHashCode<name>(); }                         \
        std::string TypeName() const override { return STypeName(); }                       \
        size_t HashCode() const override { return SHashCode(); }                            \
        bool operator==(const stratus::Entity2Component * other) const override {           \
            if (*this == other) return true;                                                \
            if (!other) return false;                                                       \
            return TypeName() == std::string(other->TypeName());                            \
        }                                                                                   

namespace stratus {
    // Meant to store any relevant data that will be manipulated by the engine + application
    // at runtime
    struct Entity2Component {
        virtual ~Entity2Component() = default;
        virtual std::string TypeName() const = 0;
        virtual size_t HashCode() const = 0;
        virtual bool operator==(const Entity2Component *) const = 0;

        void MarkChanged();
        bool ChangedWithinLastFrame() const;

        void SetEnabled(const bool);
        bool IsEnabled() const;

    protected:
        // Last engine frame this component was modified
        uint64_t _lastFrameChanged = 0;
        // Whether component should be used or not
        bool _enabled = true;
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
    };

    template<typename Component, typename ... Types>
    std::unique_ptr<EntityComponentPointerManager> ConstructComponent(Types ... args) {
        static std::mutex m;
        static PoolAllocator<Component> allocator;

        struct _Pointer : public EntityComponentPointerManager {
            _Pointer(Entity2Component * c)
                : EntityComponentPointerManager(c) {}

            virtual ~_Pointer() {
                auto ul = std::unique_lock<std::mutex>(m);
                allocator.Deallocate(dynamic_cast<Component *>(component));
            }
        };

        auto ul = std::unique_lock<std::mutex>(m);
        return std::unique_ptr<EntityComponentPointerManager>(new _Pointer(allocator.Allocate(std::forward<Types>(args)...)));
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
    // Keeps track of a unique set of components
    struct Entity2ComponentSet {
        friend class Entity2;

        ~Entity2ComponentSet();

        // Manipulate components - thread safe
        template<typename E, typename ... Types>
        void AttachComponent(Types ... args);

        template<typename E>
        bool ContainsComponent() const;

        template<typename E>
        E * GetComponent();

        template<typename E>
        const E * GetComponent() const;

        std::vector<Entity2Component *> GetAllComponents();
        std::vector<const Entity2Component *> GetAllComponents() const;

    private:
        template<typename E>
        E * _GetComponent() const;

        template<typename E, typename ... Types>
        void _AttachComponent(Types ... args);

        template<typename E>
        bool _ContainsComponent() const;

        void _AttachComponent(Entity2ComponentView);

        void _SetOwner(Entity2 *);

    private:
        mutable std::shared_mutex _m;
        Entity2 * _owner;
        // Component pointer managers (allocates and deallocates from shared pool)
        std::vector<std::unique_ptr<EntityComponentPointerManager>> _componentManagers;
        // List of unique components
        std::unordered_set<Entity2ComponentView> _components;
        // List of components by type name
        std::unordered_map<std::string, Entity2ComponentView> _componentTypeNames;
    };

    // Collection of unque ID + configurable component data
    class Entity2 : public std::enable_shared_from_this<Entity2> {
        friend class EntityManager;

        Entity2();

    public:
        static Entity2Ptr Create() {
            return Entity2Ptr(new Entity2());
        }

    public:
        ~Entity2();

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
        Entity2ComponentSet _components;
        Entity2WeakPtr _parent;
        std::vector<Entity2Ptr> _childNodes;
        // Keeps track of added/removed from world
        bool _partOfWorld = false;
    };

    template<typename E, typename ... Types>
    void Entity2ComponentSet::AttachComponent(Types ... args) {
        static_assert(std::is_base_of<Entity2Component, E>::value);
        auto ul = std::unique_lock<std::shared_mutex>(_m);
        return _AttachComponent<E>(std::forward<Types>(args)...);
    }

    template<typename E>
    bool Entity2ComponentSet::ContainsComponent() const {
        static_assert(std::is_base_of<Entity2Component, E>::value);
        auto sl = std::shared_lock<std::shared_mutex>(_m);
        return _ContainsComponent<E>();
    }

    template<typename E, typename ... Types>
    void Entity2ComponentSet::_AttachComponent(Types ... args) {
        if (_ContainsComponent<E>()) return;
        auto ptr = ConstructComponent<E>(std::forward<Types>(args)...);
        Entity2ComponentView view(dynamic_cast<Entity2Component *>(ptr->component));
        _componentManagers.push_back(std::move(ptr));
        _AttachComponent(view);
    }

    template<typename E>
    bool Entity2ComponentSet::_ContainsComponent() const {
        std::string name = E::STypeName();
        return _componentTypeNames.find(name) != _componentTypeNames.end();
    }

    template<typename E>
    E * Entity2ComponentSet::GetComponent() {
        return _GetComponent<E>();
    }

    template<typename E>
    const E * Entity2ComponentSet::GetComponent() const {
        return _GetComponent<const E>();
    }

    template<typename E>
    E * Entity2ComponentSet::_GetComponent() const {
        static_assert(std::is_base_of<Entity2Component, E>::value);
        auto sl = std::shared_lock<std::shared_mutex>(_m);
        std::string name = E::STypeName();
        auto it = _componentTypeNames.find(name);
        return it != _componentTypeNames.end() ? dynamic_cast<E *>(it->second.component) : nullptr;
    }
}