#pragma once

#include "StratusHandle.h"
#include <unordered_set>
#include <unordered_map>
#include <typeinfo>
#include <string>
#include <memory>
#include <shared_mutex>

namespace stratus {
    class Entity2;

    typedef Handle<Entity2> EntityHandle;
    typedef std::shared_ptr<Entity2> EntityPtr;

    #define ENTITY_COMPONENT_STRUCT(name)                                                       \
        struct name : public Entity2Component {                                                 \
            std::type_info TypeInfo() const override { return typeid(name); }                   \
            size_t HashCode() const override { return TypeInfo().hash_code(); }                 \
            bool operator==(const Entity2Component * other) const override {                    \
                if (*this == other) return true;                                                \
                if (!other) return false;                                                       \
                return std::string(TypeInfo().name()) == std::string(other->TypeInfo().name()); \
            }                                                                                   \
            
    struct Entity2Component {
        virtual ~Entity2Component() = default;
        virtual std::type_info TypeInfo() const = 0;
        virtual size_t HashCode() const = 0;
        virtual bool operator==(const Entity2Component *) const = 0;

        // Must be defined per component
        // If true it means that an Entity should never have more than one of this component type
        // attached. If false then an Entity can have many of the same component type as it wants.
        virtual bool IsComponentExclusive() const = 0;
    };

    // Enables an entity component pointer to be inserted into hash set/map
    struct Entity2ComponentView {
        Entity2Component * component = nullptr;

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
    class Entity2 {
        friend class World;

        Entity2();

    public:
        static std::shared_ptr<Entity2> Create() {
            return std::make_shared<Entity2>();
        }

    public:
        ~Entity2();

        EntityHandle GetID() const;

        // Manipulate components
        void AddOrOverwriteComponent(Entity2Component *);
        void RemoveComponent(Entity2Component *);
        void RemoveAllComponents();

        template<typename E>
        bool ContainsComponent() const {
            constexpr std::string name = typeid(E).name();
            return _componentTypeNames.find(name) != _componentTypeNames.end();
        }

        template<typename E>
        Entity2Component * GetComponent() const {
            constexpr std::string name = typeid(E).name();
            auto it = _componentTypeNames.find(name);
            return it != _componentTypeNames.end() ? it->second.component : nullptr;
        }

        // Change tracking
        bool ChangedThisFrame() const;
        bool ChangedLastFrame() const;
        void MarkAsChanged();


        bool IsInWorld() const;

    private:
        // Called by World class
        void AddToWorld();
        void RemoveFromWorld();

    private:
        mutable std::shared_mutex _m;
        // Unique id from all other objects in the engine
        EntityHandle _id;
        // List of unique components
        std::unordered_set<Entity2ComponentView> _components;
        // List of components by type name
        std::unordered_map<std::string, Entity2ComponentView> _componentTypeNames;
        // Keeps track of changes
        uint64_t _lastFrameChanged = 0;
        // Keeps track of added/removed from world
        bool _partOfWorld = false;
    };
}