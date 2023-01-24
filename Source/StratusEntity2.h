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

namespace stratus {
    #define ENTITY_COMPONENT_STRUCT(name)                                                       \
        struct name : public Entity2Component {                                                 \
            static constexpr std::type_info STypeInfo() { return typeid(name); }                \
            static constexpr size_t SHashCode() { return TypeInfo().hash_code(); }              \
            std::type_info TypeInfo() const override { STypeInfo(); }                           \
            size_t HashCode() const override { return SHashCode(); }                            \
            bool operator==(const Entity2Component * other) const override {                    \
                if (*this == other) return true;                                                \
                if (!other) return false;                                                       \
                return std::string(TypeInfo().name()) == std::string(other->TypeInfo().name()); \
            }                                                                                   \
            
    // Meant to store any relevant data that will be manipulated by the engine + application
    // at runtime
    struct Entity2Component {
        virtual ~Entity2Component() = default;
        virtual std::type_info TypeInfo() const = 0;
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
        template<typename E, typename ... Types>
        void AttachComponent(Types ... args);
        template<typename E>
        bool ContainsComponent() const;
        template<typename E>
        Entity2Component * GetComponent() const;
        std::vector<Entity2Component *> GetAllComponents() const;

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
        template<typename E, typename ... Types>
        void _AttachComponent(Types ... args);
        template<typename E>
        bool _ContainsComponent() const;
        void _AttachComponent(Entity2ComponentView);
        bool _ContainsChildNode(const Entity2Ptr&) const;

    private:
        mutable std::shared_mutex _m;
        // List of unique components
        std::unordered_set<Entity2ComponentView> _components;
        // List of components by type name
        std::unordered_map<std::string, Entity2ComponentView> _componentTypeNames;
        Entity2WeakPtr _parent;
        std::vector<Entity2Ptr> _childNodes;
        // Keeps track of added/removed from world
        bool _partOfWorld = false;
    };

    template<typename E, typename ... Types>
    void Entity2::AttachComponent(Types ... args) {
        static_assert(std::is_base_of<Entity2Component, E>::value);
        auto ul = std::unique_lock<std::shared_mutex>(_m);
        return _AttachComponent<E>(std::forward<Types>(args)...);
    }

    template<typename E>
    bool Entity2::ContainsComponent() const {
        static_assert(std::is_base_of<Entity2Component, E>::value);
        auto sl = std::shared_lock<std::shared_mutex>(_m);
        return _ContainsComponent<E>();
    }

    template<typename E, typename ... Types>
    void Entity2::_AttachComponent(Types ... args) {
        if (_ContainsComponent<E>()) return;
        Entity2ComponentView view(new E(std::forward<Types>(args)...));
        _AttachComponent(view);
    }

    template<typename E>
    bool Entity2::_ContainsComponent() const {
        constexpr std::string name = typeid(E).name();
        return _componentTypeNames.find(name) != _componentTypeNames.end();
    }

    template<typename E>
    Entity2Component * Entity2::GetComponent() const {
        static_assert(std::is_base_of<Entity2Component, E>::value);
        constexpr std::string name = typeid(E).name();
        auto it = _componentTypeNames.find(name);
        return it != _componentTypeNames.end() ? it->second.component : nullptr;
    }
}