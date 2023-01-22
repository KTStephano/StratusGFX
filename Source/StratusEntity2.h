#pragma once

#include "StratusHandle.h"
#include <unordered_set>
#include <unordered_map>
#include <typeinfo>
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

        // Manipulate components
        template<typename E, typename ... Types>
        void AttachComponent(Types ... args);
        template<typename E>
        void RemoveComponent();

        template<typename E>
        bool ContainsComponent() const;
        template<typename E>
        Entity2Component * GetComponent();

        bool IsInWorld() const;

    private:
        // Called by EntityManager class
        void _AddToWorld();
        void _RemoveFromWorld();
        void _CommitChanges();

    private:
        void _AttachComponent(Entity2ComponentView);
        void _RemoveComponent(Entity2ComponentView);
        void _RemoveComponentImmediate(Entity2ComponentView);

    private:
        mutable std::shared_mutex _m;
        // List of unique components
        std::unordered_set<Entity2ComponentView> _components;
        // List of components by type name
        std::unordered_map<std::string, Entity2ComponentView> _componentTypeNames;
        // Pending
        std::unordered_set<Entity2ComponentView> _pendingRemove;
        // Keeps track of added/removed from world
        bool _partOfWorld = false;
    };

    template<typename E, typename ... Types>
    void Entity2::AttachComponent(Types ... args) {
        static_assert(std::is_base_of<Entity2Component, E>::value);
        if (ContainsComponent<E>()) return;
        auto ul = std::unique_lock<std::shared_mutex>(_m);
        Entity2ComponentView view(new E(std::forward<Types>(args)...));
        _AttachComponent(view);
    }

    template<typename E>
    void Entity2::RemoveComponent() {
        static_assert(std::is_base_of<Entity2Component, E>::value);
        constexpr std::string name = typeid(E).name();
        auto ul = std::unique_lock<std::shared_mutex>(_m);
        auto it = _componentTypeNames.find(name);
        if (it != _componentTypeNames.end()) {
            _RemoveComponent(it->second);
        }
    }

    template<typename E>
    bool Entity2::ContainsComponent() const {
        constexpr std::string name = typeid(E).name();
        return _componentTypeNames.find(name) != _componentTypeNames.end();
    }

    template<typename E>
    Entity2Component * Entity2::GetComponent() {
        static_assert(std::is_base_of<Entity2Component, E>::value);
        constexpr std::string name = typeid(E).name();
        auto it = _componentTypeNames.find(name);
        return it != _componentTypeNames.end() ? it->second.component : nullptr;
    }
}