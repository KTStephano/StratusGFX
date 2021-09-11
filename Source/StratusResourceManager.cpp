#include "StratusResourceManager.h"
#include "StratusEngine.h"
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#define STB_IMAGE_IMPLEMENTATION
#include "STBImage.h"

    //     ResourceManager();

    // public:
    //     ResourceManager(const ResourceManager&) = delete;
    //     ResourceManager(ResourceManager&&) = delete;
    //     ResourceManager& operator=(const ResourceManager&) = delete;
    //     ResourceManager& operator=(ResourceManager&&) = delete;

    //     ~ResourceManager();

    //     static ResourceManager * Instance() { return _instance; }

    //     void Update();
    //     Async<Entity> LoadModel(const std::string&);

    // private:
    //     static ResourceManager * _instance;
    //     std::vector<ThreadPtr> _threads;
    //     std::vector<std::vector<Thread::ThreadFunction>> _resourceLoadRequests;
    //     uint32_t _nextResourceVector = 0;
    //     mutable std::shared_mutex _mutex;

namespace stratus {
    ResourceManager * ResourceManager::_instance = nullptr;

    ResourceManager::ResourceManager() {
        for (int i = 0; i < 1; ++i) {
            _threads.push_back(ThreadPtr(new Thread("StreamingThread#" + std::to_string(i + 1), true)));
        }
    }

    ResourceManager::~ResourceManager() {

    }

    void ResourceManager::Update() {
        for (auto& thread : _threads) {
            if (thread->Idle()) {
                thread->Dispatch();
            }
        }
    }

    Async<Entity> ResourceManager::LoadModel(const std::string& name) {
        {
            auto sl = _LockRead();
            if (_loadedModels.find(name) != _loadedModels.end()) {
                Async<Entity> e = _loadedModels.find(name)->second;
                return (e.Completed() && !e.Failed()) ? Async<Entity>(e.GetPtr()->Copy()) : e;
            }
        }

        auto ul = _LockWrite();
        auto index = _NextResourceIndex();
        Async<Entity> e(*_threads[index].get(), [this, name]() {
            return _LoadModel(name);
        });

        _loadedModels.insert(std::make_pair(name, e));
        return e;
    }

    uint32_t ResourceManager::_NextResourceIndex() {
        uint32_t index = _nextResourceVector % _threads.size();
        ++_nextResourceVector;
        return index;
    }

    TextureHandle ResourceManager::LoadTexture(const std::string& name) {
        {
            auto sl = _LockRead();
            if (_loadedTexturesByFile.find(name) != _loadedTexturesByFile.end()) {
                return _loadedTexturesByFile.find(name)->second;
            }
        }

        auto ul = _LockWrite();
        auto index = _NextResourceIndex();
        auto handle = TextureHandle::NextHandle();
        // We have to use the main thread since Texture calls glGenTextures :(
        Async<Texture> as(*Engine::Instance()->GetMainThread(), [this, name, handle]() {
            return _LoadTexture(name, handle);
        });

        _loadedTexturesByFile.insert(std::make_pair(name, handle));
        _loadedTextures.insert(std::make_pair(handle, as));

        return handle;
    }

    bool ResourceManager::GetTexture(const TextureHandle handle, Async<Texture>& tex) const {
        auto sl = _LockRead();
        if (_loadedTextures.find(handle) == _loadedTextures.end()) {
            return false; // not loaded
        }

        tex = _loadedTextures.find(handle)->second;
        return true;
    }

    Entity * ResourceManager::_LoadModel(const std::string& name) const {
        Assimp::Importer importer;
        //const aiScene *scene = importer.ReadFile(filename, aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_CalcTangentSpace | aiProcess_GenNormals | aiProcess_GenUVCoords);
        //const aiScene *scene = importer.ReadFile(filename, aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_FlipUVs | aiProcess_CalcTangentSpace | aiProcess_OptimizeMeshes);
        const aiScene *scene = importer.ReadFile(name, aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_FlipUVs | aiProcess_GenUVCoords | aiProcess_CalcTangentSpace | aiProcess_SplitLargeMeshes);

        auto material = MaterialManager::Instance()->CreateMaterial(name);

        if(!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
            STRATUS_ERROR << "Error loading model: " << name << std::endl << importer.GetErrorString() << std::endl;
            return nullptr;
        }

        Entity * e = new Entity();
        const std::string directory = name.substr(0, name.find_last_of('/'));
        //processNode(scene->mRootNode, scene, e, material, directory);
        return e;
    }

    Texture * ResourceManager::_LoadTexture(const std::string& file, const TextureHandle handle) const {
        STRATUS_LOG << "Attempting to load texture from file: " << file << std::endl;
        Texture * texture;
        int width, height, numChannels;
        // @see http://www.redbancosdealimentos.org/homes-flooring-design-sources
        uint8_t * data = stbi_load(file.c_str(), &width, &height, &numChannels, 0);
        if (data) {
            TextureConfig config;
            config.type = TextureType::TEXTURE_2D;
            config.storage = TextureComponentSize::BITS_DEFAULT;
            config.generateMipMaps = true;
            config.dataType = TextureComponentType::UINT;
            config.width = (uint32_t)width;
            config.height = (uint32_t)height;
            // This loads the textures with sRGB in mind so that they get converted back
            // to linear color space. Warning: if the texture was not actually specified as an
            // sRGB texture (common for normal/specular maps), this will cause problems.
            switch (numChannels) {
                case 1:
                    config.format = TextureComponentFormat::RED;
                    break;
                case 3:
                    config.format = TextureComponentFormat::SRGB;
                    break;
                case 4:
                    config.format = TextureComponentFormat::SRGB_ALPHA;
                    break;
                default:
                    STRATUS_ERROR << "Unknown texture loading error - format may be invalid" << std::endl;
                    stbi_image_free(data);
                    return nullptr;
            }

            texture = new Texture(config, data, false);
            texture->_setHandle(handle);
            texture->setCoordinateWrapping(TextureCoordinateWrapping::REPEAT);
            texture->setMinMagFilter(TextureMinificationFilter::LINEAR_MIPMAP_LINEAR, TextureMagnificationFilter::LINEAR);
        } 
        else {
            STRATUS_ERROR << "Could not load texture: " << file << std::endl;
            return nullptr;
        }
        
        stbi_image_free(data);
        return texture;
    }
}