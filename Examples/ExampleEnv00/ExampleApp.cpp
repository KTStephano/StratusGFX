#include "StratusEngine.h"
#include "StratusLog.h"

class ExampleApp : public stratus::Application {
public:
    virtual ~ExampleApp() = default;

    std::string GetAppName() const {
        return "ExampleApp00";
    }

    virtual bool Initialize() override {
        return true; // success
    }

    virtual stratus::SystemStatus Update(double deltaSeconds) override {
        STRATUS_LOG << "Successfully entered ExampleApp::Update! Delta seconds = " << deltaSeconds << std::endl;
        return stratus::SystemStatus::SYSTEM_SHUTDOWN;
    }

    virtual void ShutDown() override {
        STRATUS_LOG << "Successfully entered ExampleApp::ShutDown()" << std::endl;
    }
};

STRATUS_ENTRY_POINT(ExampleApp)