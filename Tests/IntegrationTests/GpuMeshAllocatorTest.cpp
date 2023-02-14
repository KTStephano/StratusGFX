#include <catch2/catch_all.hpp>
#include <iostream>
#include <unordered_set>

#include "StratusEngine.h"
#include "StratusApplication.h"
#include "StratusLog.h"
#include "StratusCommon.h"
#include "IntegrationMain.h"
#include "StratusGpuBuffer.h"

TEST_CASE( "Stratus GpuMeshAllocator Test", "[stratus_gpu_mesh_allocator_test]" ) {
    static bool failed;
    failed = false;

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

            // Make sure basic allocate + free works
            auto freeVertices = stratus::GpuMeshAllocator::FreeVertices();
            auto vertexOffset = stratus::GpuMeshAllocator::AllocateVertexData(freeVertices);
            stratus::GpuMeshAllocator::DeallocateVertexData(vertexOffset, freeVertices);

            if (freeVertices != stratus::GpuMeshAllocator::FreeVertices()) {
                failed = true;
            }

            if (vertexOffset != stratus::GpuMeshAllocator::AllocateVertexData(8)) {
                failed = true;
            }
            stratus::GpuMeshAllocator::DeallocateVertexData(vertexOffset, 8);

            // Make sure advanced allocate + free works
            vertexOffset = stratus::GpuMeshAllocator::AllocateVertexData(freeVertices);
            STRATUS_LOG << stratus::GpuMeshAllocator::FreeVertices() << std::endl;

            if (stratus::GpuMeshAllocator::FreeVertices() != 0) {
                failed = true;
            }

            stratus::GpuMeshAllocator::DeallocateVertexData(vertexOffset, 1);

            if (vertexOffset != stratus::GpuMeshAllocator::AllocateVertexData(1)) {
                failed = true;
            }

            // Make sure merge works
            stratus::GpuMeshAllocator::DeallocateVertexData(vertexOffset, 1);
            stratus::GpuMeshAllocator::DeallocateVertexData(vertexOffset + 1, 1);
            stratus::GpuMeshAllocator::DeallocateVertexData(vertexOffset + 2, 1);
            if (vertexOffset != stratus::GpuMeshAllocator::AllocateVertexData(3)) {
                failed = true;
            }

            stratus::GpuMeshAllocator::DeallocateVertexData(vertexOffset + 20, 1);
            stratus::GpuMeshAllocator::DeallocateVertexData(vertexOffset + 21, 1);
            stratus::GpuMeshAllocator::DeallocateVertexData(vertexOffset + 22, 1);
            if ((vertexOffset + 20) != stratus::GpuMeshAllocator::AllocateVertexData(3)) {
                failed = true;
            }

            if (stratus::GpuMeshAllocator::FreeVertices() != 0) {
                failed = true;
            }

            // Deallocate in many random places
            srand(time(NULL));
            std::vector<uint32_t> offsets;
            std::unordered_set<uint32_t> deallocated;
            for (int i = 0; i < 250; ++i) {
                // freeVertices - 1 to prevent the allocator from merging the individual chunks
                // back into the main list in any circumstance
                uint32_t location = vertexOffset + i + 1;
                if (deallocated.find(location) != deallocated.end()) {
                    --i;
                    continue;
                }
                deallocated.insert(location);
                offsets.push_back(location);
                stratus::GpuMeshAllocator::DeallocateVertexData(location, 1);
            }

            std::sort(offsets.begin(), offsets.end());
            for (int i = 0; i < offsets.size(); ++i) {
                auto offset = stratus::GpuMeshAllocator::AllocateVertexData(1);
                if (offsets[i] != offset) {
                    failed = true;
                }
            }

            return stratus::SystemStatus::SYSTEM_SHUTDOWN;
        }

        virtual void Shutdown() override {
            STRATUS_LOG << "Successfully entered GpuMeshAllocatorTest::ShutDown()" << std::endl;
        }
    };

    STRATUS_INLINE_ENTRY_POINT(GpuMeshAllocatorTest, numArgs, argList);

    REQUIRE_FALSE(failed);
}