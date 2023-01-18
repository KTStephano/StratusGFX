#pragma once

#include "StratusSystemModule.h"
#include <string>

namespace stratus {
    class ResourceManager;
    class RendererFrontend;
    class MaterialManager;
    class Engine;
    class Window;

    // Special interface class which the engine knows is the entry point
    // for the application (e.g. editor or game)
    class Application : public SystemModule {
        friend class Engine;

    public:
        static Application * Instance() { return _instance; }

        virtual ~Application() = default;

        // Sets the name of the window
        virtual const char * GetAppName() const = 0;

        virtual const char * Name() const {
            return GetAppName();
        }

        // Convenience functions for common use cases
        static ResourceManager * Resources();
        static RendererFrontend * World();
        static MaterialManager * Materials();
        static Engine * Engine();
        static Window * Window();

    private:
        static Application * _instance;
    };
}