#include "StratusCommon.h"
#include "glm/glm.hpp"
#include <iostream>
#include <StratusPipeline.h>
#include <StratusCamera.h>
#include <chrono>
#include "StratusEngine.h"
#include "StratusResourceManager.h"
#include "StratusLog.h"
#include "StratusRendererFrontend.h"
#include "StratusWindow.h"
#include <StratusLight.h>
#include <StratusUtils.h>
#include <memory>
#include <filesystem>
#include "CameraController.h"
#include "WorldLightController.h"
#include "LightComponents.h"
#include "LightControllers.h"
#include "StratusTransformComponent.h"
#include "StratusGpuCommon.h"
#include "WorldLightController.h"
#include "FrameRateController.h"
#include "StratusGraphicsDriver.h"
#include "StratusPipeline.h"

static const std::vector<GLfloat> cubeData = std::vector<GLfloat>{
    // back face
    // positions          // normals          // tex coords     // tangent   // bitangent
    -1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 0.0f, 0.0f,       1, 0, 0,     0, 1, 0, // bottom-left
    1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 1.0f, 1.0f,        1, 0, 0,     0, 1, 0,// top-right
    1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 1.0f, 0.0f,        1, 0, 0,     0, 1, 0,// bottom-right
    1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 1.0f, 1.0f,        1, 0, 0,     0, 1, 0, // top-right
    -1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 0.0f, 0.0f,       1, 0, 0,     0, 1, 0, // bottom-left
    -1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 0.0f, 1.0f,       1, 0, 0,     0, 1, 0, // top-left
    // front face        
    -1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f, 0.0f,       1, 0, 0,     0, 1, 0, // bottom-left
    1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f, 0.0f,        1, 0, 0,     0, 1, 0,// bottom-right
    1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f, 1.0f,        1, 0, 0,     0, 1, 0,// top-right
    1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f, 1.0f,        1, 0, 0,     0, 1, 0, // top-right
    -1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f, 1.0f,       1, 0, 0,     0, 1, 0, // top-left
    -1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f, 0.0f,       1, 0, 0,     0, 1, 0, // bottom-left
    // left face        
    -1.0f,  1.0f,  1.0f, -1.0f,  0.0f,  0.0f, 1.0f, 0.0f,       0, 1, 0,     0, 0, -1, // top-right
    -1.0f,  1.0f, -1.0f, -1.0f,  0.0f,  0.0f, 1.0f, 1.0f,       0, 1, 0,     0, 0, -1,// top-left
    -1.0f, -1.0f, -1.0f, -1.0f,  0.0f,  0.0f, 0.0f, 1.0f,       0, 1, 0,     0, 0, -1,// bottom-left
    -1.0f, -1.0f, -1.0f, -1.0f,  0.0f,  0.0f, 0.0f, 1.0f,       0, 1, 0,     0, 0, -1, // bottom-left
    -1.0f, -1.0f,  1.0f, -1.0f,  0.0f,  0.0f, 0.0f, 0.0f,       0, 1, 0,     0, 0, -1, // bottom-right
    -1.0f,  1.0f,  1.0f, -1.0f,  0.0f,  0.0f, 1.0f, 0.0f,       0, 1, 0,     0, 0, -1,// top-right
    // right face        
    1.0f,  1.0f,  1.0f,  1.0f,  0.0f,  0.0f, 1.0f, 0.0f,        0, 1, 0,     0, 0, -1,// top-left
    1.0f, -1.0f, -1.0f,  1.0f,  0.0f,  0.0f, 0.0f, 1.0f,        0, 1, 0,     0, 0, -1, // bottom-right
    1.0f,  1.0f, -1.0f,  1.0f,  0.0f,  0.0f, 1.0f, 1.0f,        0, 1, 0,     0, 0, -1,// top-right
    1.0f, -1.0f, -1.0f,  1.0f,  0.0f,  0.0f, 0.0f, 1.0f,        0, 1, 0,     0, 0, -1, // bottom-right
    1.0f,  1.0f,  1.0f,  1.0f,  0.0f,  0.0f, 1.0f, 0.0f,        0, 1, 0,     0, 0, -1, // top-left
    1.0f, -1.0f,  1.0f,  1.0f,  0.0f,  0.0f, 0.0f, 0.0f,        0, 1, 0,     0, 0, -1,// bottom-left
    // bottom face        
    -1.0f, -1.0f, -1.0f,  0.0f, -1.0f,  0.0f, 0.0f, 1.0f,       1, 0, 0,     0, 0, -1,// top-right
    1.0f, -1.0f, -1.0f,  0.0f, -1.0f,  0.0f, 1.0f, 1.0f,        1, 0, 0,     0, 0, -1,// top-left
    1.0f, -1.0f,  1.0f,  0.0f, -1.0f,  0.0f, 1.0f, 0.0f,        1, 0, 0,     0, 0, -1,// bottom-left
    1.0f, -1.0f,  1.0f,  0.0f, -1.0f,  0.0f, 1.0f, 0.0f,        1, 0, 0,     0, 0, -1,// bottom-left
    -1.0f, -1.0f,  1.0f,  0.0f, -1.0f,  0.0f, 0.0f, 0.0f,       1, 0, 0,     0, 0, -1,// bottom-right
    -1.0f, -1.0f, -1.0f,  0.0f, -1.0f,  0.0f, 0.0f, 1.0f,       1, 0, 0,     0, 0, -1,// top-right
    // top face        
    -1.0f,  1.0f, -1.0f,  0.0f,  1.0f,  0.0f, 0.0f, 1.0f,       1, 0, 0,     0, 0, -1,// top-left
    1.0f,  1.0f , 1.0f,  0.0f,  1.0f,  0.0f, 1.0f, 0.0f,        1, 0, 0,     0, 0, -1,// bottom-right
    1.0f,  1.0f, -1.0f,  0.0f,  1.0f,  0.0f, 1.0f, 1.0f,        1, 0, 0,     0, 0, -1,// top-right
    1.0f,  1.0f,  1.0f,  0.0f,  1.0f,  0.0f, 1.0f, 0.0f,        1, 0, 0,     0, 0, -1,// bottom-right
    -1.0f,  1.0f, -1.0f,  0.0f,  1.0f,  0.0f, 0.0f, 1.0f,       1, 0, 0,     0, 0, -1,// top-left
    -1.0f,  1.0f,  1.0f,  0.0f,  1.0f,  0.0f, 0.0f, 0.0f,       1, 0, 0,     0, 0, -1// bottom-left
};

struct VertexData {
    float position[3];
    float uv[2];
    float normal[3];
};

class Custom : public stratus::Application {
public:
    virtual ~Custom() = default;

    const char * GetAppName() const override {
        return "Custom";
    }

    // Perform first-time initialization - true if success, false otherwise
    virtual bool Initialize() override {
        STRATUS_LOG << "Initializing " << GetAppName() << std::endl;

        srand(time(nullptr));
        
        stratus::InputHandlerPtr controller(new CameraController());
        Input()->AddInputHandler(controller);

        INSTANCE(RendererFrontend)->SetEnableRenderingLoop(false);

        const std::filesystem::path shaderRoot("../Examples/ExampleEnv08/");
        const stratus::ShaderApiVersion version{ 
            stratus::GraphicsDriver::GetConfig().majorVersion, 
            stratus::GraphicsDriver::GetConfig().minorVersion };

        bindless = std::unique_ptr<stratus::Pipeline>(new stratus::Pipeline(shaderRoot, version, {
            stratus::Shader{"bindless.vs", stratus::ShaderType::VERTEX},
            stratus::Shader{"bindless.fs", stratus::ShaderType::FRAGMENT} }
        ));

        std::vector<VertexData> data;
        for (size_t i = 0; i < cubeData.size(); i += 14) {
            VertexData v;
            v.position[0] = cubeData[i];
            v.position[1] = cubeData[i + 1];
            v.position[2] = cubeData[i + 2];

            v.normal[0] = cubeData[i + 3];
            v.normal[1] = cubeData[i + 4];
            v.normal[2] = cubeData[i + 5];

            v.uv[0] = cubeData[i + 6];
            v.uv[1] = cubeData[i + 7];

            data.push_back(std::move(v));
        }

        numVertices = data.size();

        glCreateBuffers(1, &verticesBuffer);
        glNamedBufferStorage(
            verticesBuffer,
            sizeof(VertexData) * data.size(),
            (const void*)data.data(),
            GL_DYNAMIC_STORAGE_BIT
        );

        std::vector<glm::mat4> instancedMatrices;

        for (size_t x = 0; x < 200; x += 5) {
            for (size_t y = 0; y < 200; y += 5) {
                glm::mat4 mat(1.0f);
                stratus::matTranslate(mat, glm::vec3(float(x), float(y), 0.0f));
                instancedMatrices.push_back(std::move(mat));
            }
        }

        numInstances = instancedMatrices.size();

        glCreateBuffers(1, &modelMatrices);
        glNamedBufferStorage(
            modelMatrices,
            sizeof(glm::mat4) * instancedMatrices.size(),
            (const void*)instancedMatrices.data(),
            GL_DYNAMIC_STORAGE_BIT
        );

        std::vector<GLuint> textures;
        const size_t textureSize = 32 * 32 * 3;
        unsigned char textureData[textureSize];
        for (int i = 0; i < numInstances; ++i) {
            const unsigned char limit = unsigned char(rand() % 231 + 25);
            for (int j = 0; j < textureSize; ++j) {
                textureData[j] = unsigned char(rand() % limit);
            }

            GLuint texture;
            glCreateTextures(GL_TEXTURE_2D, 1, &texture);
            glTextureStorage2D(texture, 1, GL_RGB8, 32, 32);
            glTextureSubImage2D(texture, 0, 0, 0, 32, 32, GL_RGB, GL_UNSIGNED_BYTE, (const void *)&textureData[0]);
            glGenerateTextureMipmap(texture);

            const GLuint64 handle = glGetTextureHandleARB(texture);
            STRATUS_LOG << handle << std::endl;
            textureHandles.push_back(handle);
        }

        glCreateBuffers(1, &textureBuffer);
        glNamedBufferStorage(
            textureBuffer,
            sizeof(GLuint64) * textureHandles.size(),
            (const void *)textureHandles.data(),
            GL_DYNAMIC_STORAGE_BIT
        );

        if (!bindless->isValid()) {
            return false;
        }

        STRATUS_LOG << std::filesystem::current_path() << std::endl;

        return true;
    }

    // Run a single update for the application (no infinite loops)
    // deltaSeconds = time since last frame
    virtual stratus::SystemStatus Update(const double deltaSeconds) override {
        if (Engine()->FrameCount() % 100 == 0) {
            STRATUS_LOG << "FPS:" << (1.0 / deltaSeconds) << " (" << (deltaSeconds * 1000.0) << " ms)" << std::endl;
        }

        //STRATUS_LOG << "Camera " << camera.getYaw() << " " << camera.getPitch() << std::endl;

        auto camera = World()->GetCamera();

        // Check for key/mouse events
        auto events = Input()->GetInputEventsLastFrame();
        for (auto e : events) {
            switch (e.type) {
                case SDL_QUIT:
                    return stratus::SystemStatus::SYSTEM_SHUTDOWN;
                case SDL_KEYDOWN:
                case SDL_KEYUP: {
                    bool released = e.type == SDL_KEYUP;
                    SDL_Scancode key = e.key.keysym.scancode;
                    switch (key) {
                        case SDL_SCANCODE_ESCAPE:
                            if (released) {
                                return stratus::SystemStatus::SYSTEM_SHUTDOWN;
                            }
                            break;
                    }
                    break;
                }
                default: break;
            }
        }

        stratus::GraphicsDriver::MakeContextCurrent();

        glEnable(GL_DEPTH_TEST);
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Mark all as resident
        for (GLuint64 handle : textureHandles) {
            glMakeTextureHandleResidentARB(handle);
        }

        bindless->bind();

        bindless->setMat4("view", camera->getViewTransform());
        bindless->setMat4("projection", INSTANCE(RendererFrontend)->GetProjectionMatrix());

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, verticesBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, modelMatrices);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, textureBuffer);

        glDrawArraysInstanced(GL_TRIANGLES, 0, numVertices, numInstances);

        bindless->unbind();

        // Mark all as non-resident
        for (GLuint64 handle : textureHandles) {
            glMakeTextureHandleNonResidentARB(handle);
        }

        stratus::GraphicsDriver::SwapBuffers(true);

        return stratus::SystemStatus::SYSTEM_CONTINUE;
    }

    // Perform any resource cleanup
    virtual void Shutdown() override {
        glDeleteBuffers(1, &verticesBuffer);
    }

    std::unique_ptr<stratus::Pipeline> bindless;
    int numVertices = 0;
    int numInstances = 0;
    std::vector<GLuint64> textureHandles;
    GLuint verticesBuffer;
    GLuint modelMatrices;
    GLuint textureBuffer;
};

STRATUS_ENTRY_POINT(Custom)