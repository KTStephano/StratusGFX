#include "StratusApplication.h"
#include "StratusResourceManager.h"
#include "StratusRendererFrontend.h"
#include "StratusMaterial.h"
#include "StratusEngine.h"
#include "StratusWindow.h"
#include "StratusCommon.h"

namespace stratus {
    Application * Application::_instance = nullptr;

    ResourceManager * Application::Resources() {
        return INSTANCE(ResourceManager);
    }

    RendererFrontend * Application::World() {
        return INSTANCE(RendererFrontend);
    }

    MaterialManager * Application::Materials() {
        return INSTANCE(MaterialManager);
    }

    Engine * Application::Engine() {
        return INSTANCE(Engine);
    }

    InputManager * Application::Input() {
        return INSTANCE(InputManager);
    }

    Window * Application::Window() {
        return INSTANCE(Window);
    }
}