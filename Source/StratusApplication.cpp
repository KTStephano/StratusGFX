#include "StratusApplication.h"
#include "StratusResourceManager.h"
#include "StratusRendererFrontend.h"
#include "StratusMaterial.h"
#include "StratusEngine.h"

namespace stratus {
    #define INSTANCE(type) stratus::##type::Instance()

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

    Engine * Application::Stratus() {
        return INSTANCE(Engine);
    }
}