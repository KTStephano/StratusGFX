#include <catch2/catch_all.hpp>
#include <iostream>
#include <unordered_set>

#include "StratusEngine.h"
#include "StratusApplication.h"
#include "StratusLog.h"
#include "StratusCommon.h"
#include "IntegrationMain.h"

TEST_CASE( "Stratus GpuMeshAllocator Test", "[stratus_gpu_mesh_allocator_test]" ) {
    class GpuMeshAllocatorTest : public stratus::Application {
    public:
        virtual ~GpuMeshAllocatorTest() = default;

        const char * GetAppName() const override {
            return "GpuMeshAllocatorTest";
        }

        virtual bool Initialize() override {
            return true; // success
        }

        virtual stratus::SystemStatus Update(const double deltaSeconds) override {
            STRATUS_LOG << "Successfully entered GpuMeshAllocatorTest::Update! Delta seconds = " << deltaSeconds << std::endl;
            return stratus::SystemStatus::SYSTEM_SHUTDOWN;
        }

        virtual void Shutdown() override {
            STRATUS_LOG << "Successfully entered GpuMeshAllocatorTest::ShutDown()" << std::endl;
        }
    };

    STRATUS_INLINE_ENTRY_POINT(GpuMeshAllocatorTest, numArgs, argList);
}