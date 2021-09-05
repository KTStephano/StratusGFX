#pragma once

#include "StratusHandle.h"
#include <memory>
#include <unordered_set>
#include <shared_mutex>
#include <cstdint>
#include <string>

namespace stratus {
    class Scene;
    class SceneManager;
    
    typedef std::shared_ptr<Scene> ScenePtr;

    // A scene is a grouping of entities and their resources. When
    // a scene is active (see SceneManager), its entities are updated each frame
    // and rendered on the screen as needed.
    class Scene {
        friend class SceneManager;

        Scene(const std::string& name);

    public:

    };

    // Scene manager keeps track of all scenes that currently exist. In addition,
    // any currently-active scene is represented here for updates + rendering.
    struct SceneManager {
        friend class Engine;

        SceneManager();
    
    public:
        ~SceneManager();

        // Gets global instance
        static SceneManager * Instance() { return _instance; }

        // Functions for creating, getting and manipulating scenes
        ScenePtr CreateScene(const std::string&);
        void DeleteScene(const std::string&);
        bool IsSceneNameAvailable(const std::string&) const;
        void SetCurrentActiveScene(const std::string&);

        ScenePtr GetScene(const std::string&) const;
        const std::vector<ScenePtr>& GetAllScenes() const;

        // Should be called by the engine each frame
        void Update(double deltaSeconds);

    private:
        static SceneManager * _instance;
        // Keeps track of all the names currently in use
        std::unordered_set<std::string> _sceneNames;
        // Protects critical section
        mutable std::shared_mutex _mutex;
    };
}