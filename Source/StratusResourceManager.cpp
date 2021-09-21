#include "StratusResourceManager.h"
#include "StratusEngine.h"
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include "StratusRendererFrontend.h"
#include <sstream>
#define STB_IMAGE_IMPLEMENTATION
#include "STBImage.h"

namespace stratus {
    ResourceManager * ResourceManager::_instance = nullptr;

    ResourceManager::ResourceManager() {
        for (int i = 0; i < 8; ++i) {
            _threads.push_back(ThreadPtr(new Thread("StreamingThread#" + std::to_string(i + 1), true)));
        }

        _InitCube();
        _InitQuad();
    }

    ResourceManager::~ResourceManager() {

    }

    void ResourceManager::Update() {
        {
            auto ul = _LockWrite();
            _ClearAsyncTextureData();
            _ClearAsyncModelData();
        }

        for (auto& thread : _threads) {
            if (thread->Idle()) {
                thread->Dispatch();
            }
        }
    }

    void ResourceManager::_ClearAsyncTextureData() {
        std::vector<TextureHandle> toDelete;
        constexpr size_t maxBytes = 1024 * 1024 * 10; // 10 mb per frame
        size_t totalTex = 0;
        size_t totalBytes = 0;
        std::vector<TextureHandle> handles;
        std::vector<std::shared_ptr<RawTextureData>> rawTexData;
        for (auto& tpair : _asyncLoadedTextureData) {
            if (totalBytes > maxBytes) break;
            if (tpair.second.Completed()) {
                toDelete.push_back(tpair.first);
                TextureHandle handle = tpair.first;
                auto texdata = tpair.second.GetPtr();
                totalBytes += texdata->sizeBytes;
                ++totalTex;

                handles.push_back(handle);
                rawTexData.push_back(texdata);
            }
        }

        if (totalBytes > 0) {
            STRATUS_LOG << "Texture data bytes processed: " << totalBytes << ", " << totalTex << std::endl;
        }

        for (auto handle : toDelete) _asyncLoadedTextureData.erase(handle);

        RendererFrontend::Instance()->QueueRendererThreadTask([this, handles, rawTexData]() {
            std::vector<Texture *> ptrs(rawTexData.size());
            for (int i = 0; i < rawTexData.size(); ++i) {
                ptrs[i] = _FinalizeTexture(*rawTexData[i]);
            }

            auto ul = _LockWrite();
            for (int i = 0; i < handles.size(); ++i) {
                _loadedTextures.insert(std::make_pair(handles[i], Async<Texture>(std::shared_ptr<Texture>(ptrs[i]))));
            }
        });
    }

    void ResourceManager::_ClearAsyncModelData() {
        constexpr size_t maxBytes = 1024 * 1024 * 10; // 10 mb per frame
        std::vector<std::string> toDelete;
        for (auto& mpair : _pendingFinalize) {
            if (mpair.second.Completed() && !mpair.second.Failed()) {
                toDelete.push_back(mpair.first);
                auto ptr = mpair.second.GetPtr();

                _ClearAsyncModelData(ptr);
           }
        }

        for (const std::string& name : toDelete) {
           _pendingFinalize.erase(name);
        }

        size_t totalBytes = 0;
        std::vector<Thread::ThreadFunction> functions;
        std::vector<RenderMeshPtr> meshesToDelete;

        for (RenderMeshPtr mesh : _meshFinalizeQueue) {
            totalBytes += mesh->GetGpuSizeBytes();
            functions.push_back([this, mesh]() {
                mesh->GenerateGpuData();
                FinalizeModelMemory(mesh);
            });
            meshesToDelete.push_back(mesh);
            if (totalBytes >= maxBytes) break;
        }

        RendererFrontend::Instance()->QueueRendererThreadTasks(functions);

        for (RenderMeshPtr mesh : meshesToDelete) _meshFinalizeQueue.erase(mesh);

        if (totalBytes > 0) STRATUS_LOG << "Processed " << totalBytes << " bytes of mesh data: " << meshesToDelete.size() << " meshes" << std::endl;
    }

    void ResourceManager::_ClearAsyncModelData(EntityPtr ptr) {
        if (ptr == nullptr) return;
        for (auto& child : ptr->GetChildren()) {
            _ClearAsyncModelData(child);
        }

        auto rnode = ptr->GetRenderNode();
        if (rnode == nullptr || rnode->GetNumMeshContainers() == 0) return;

        for (int i = 0; i < rnode->GetNumMeshContainers(); ++i) {
            _meshFinalizeQueue.insert(rnode->GetMeshContainer(i)->mesh);
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
        _pendingFinalize.insert(std::make_pair(name, e));
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
        Async<RawTextureData> as(*_threads[index].get(), [this, name, handle]() {
            return _LoadTexture(name, handle);
        });

        _loadedTexturesByFile.insert(std::make_pair(name, handle));
        _asyncLoadedTextureData.insert(std::make_pair(handle, as));

        return handle;
    }

    void ResourceManager::FinalizeModelMemory(const RenderMeshPtr& ptr) {
        auto sl = _LockWrite();
        auto index = _NextResourceIndex();
        Async<bool> as(*_threads[index].get(), [ptr]() {
            ptr->FinalizeGpuData();
            return (bool *)nullptr;
        });
    }

    bool ResourceManager::GetTexture(const TextureHandle handle, Async<Texture>& tex) const {
        auto sl = _LockRead();
        if (_loadedTextures.find(handle) == _loadedTextures.end()) {
            return false; // not loaded
        }

        tex = _loadedTextures.find(handle)->second;
        return true;
    }

    static TextureHandle LoadMaterialTexture(aiMaterial * mat, const aiTextureType& type, const std::string& directory) {
        TextureHandle texture;
        if (mat->GetTextureCount(type) > 0) {
            aiString str; 
            mat->GetTexture(type, 0, &str);
            std::string file = str.C_Str();
            texture = ResourceManager::Instance()->LoadTexture(directory + "/" + file);
        }

        return texture;
    }

    static void PrintMatType(const aiMaterial * aimat, const aiTextureType type) {
        const auto count = aimat->GetTextureCount(type);
        std::stringstream out;
        out << count;
        if (count > 0) {
            aiString str;
            aimat->GetTexture(type, 0, &str);
            std::string file = str.C_Str();
            out << ", " << file;
        }
        out << std::endl;
        STRATUS_LOG << out.str();
    }

    static void ProcessMesh(RenderNodePtr renderNode, aiMesh * mesh, const aiScene * scene, MaterialPtr rootMat, const std::string& directory) {
        if (mesh->mNumUVComponents[0] == 0) return;

        RenderMeshPtr rmesh = RenderMeshPtr(new RenderMesh());
        // Process core primitive data
        for (uint32_t i = 0; i < mesh->mNumVertices; i++) {
            rmesh->AddVertex(glm::vec3(mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z));
            rmesh->AddUV(glm::vec2(mesh->mTextureCoords[0][i].x, mesh->mTextureCoords[0][i].y));
            rmesh->AddNormal(glm::vec3(mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z));
            rmesh->AddTangent(glm::vec3(mesh->mTangents[i].x, mesh->mTangents[i].y, mesh->mTangents[i].z));
            rmesh->AddBitangent(glm::vec3(mesh->mBitangents[i].x, mesh->mBitangents[i].y, mesh->mBitangents[i].z));
        }

        // Process indices
        for(uint32_t i = 0; i < mesh->mNumFaces; i++) {
            aiFace face = mesh->mFaces[i];
            for(uint32_t j = 0; j < face.mNumIndices; j++) {
                rmesh->AddIndex(face.mIndices[j]);
            }
        }

        // Process material
        MaterialPtr m = rootMat->CreateSubMaterial();
        if (mesh->mMaterialIndex >= 0) {
            aiMaterial * aimat = scene->mMaterials[mesh->mMaterialIndex];

            STRATUS_LOG << "mat\n";
            PrintMatType(aimat, aiTextureType_DIFFUSE);
            PrintMatType(aimat, aiTextureType_SPECULAR);
            PrintMatType(aimat, aiTextureType_AMBIENT);
            PrintMatType(aimat, aiTextureType_EMISSIVE);
            PrintMatType(aimat, aiTextureType_HEIGHT);
            PrintMatType(aimat, aiTextureType_NORMALS);
            PrintMatType(aimat, aiTextureType_SHININESS);
            PrintMatType(aimat, aiTextureType_OPACITY);
            PrintMatType(aimat, aiTextureType_DISPLACEMENT);
            PrintMatType(aimat, aiTextureType_LIGHTMAP);
            PrintMatType(aimat, aiTextureType_REFLECTION);

            m->SetDiffuseTexture(LoadMaterialTexture(aimat, aiTextureType_DIFFUSE, directory));
            m->SetNormalMap(LoadMaterialTexture(aimat, aiTextureType_NORMALS, directory));
            m->SetDepthMap(LoadMaterialTexture(aimat, aiTextureType_HEIGHT, directory));
            m->SetRoughnessMap(LoadMaterialTexture(aimat, aiTextureType_SHININESS, directory));
            m->SetAmbientTexture(LoadMaterialTexture(aimat, aiTextureType_AMBIENT, directory));
            m->SetMetallicMap(LoadMaterialTexture(aimat, aiTextureType_SPECULAR, directory));
            STRATUS_LOG << "m " 
                << m->GetDiffuseTexture() << " "
                << m->GetNormalMap() << " "
                << m->GetDepthMap() << " "
                << m->GetRoughnessMap() << " "
                << m->GetAmbientTexture() << " "
                << m->GetMetallicMap() << std::endl;
        }

        rmesh->GenerateCpuData();
        renderNode->AddMeshContainer(RenderMeshContainer{rmesh, m});
        renderNode->SetFaceCullMode(RenderFaceCulling::CULLING_CCW);
    }

    static void ProcessNode(aiNode * node, const aiScene * scene, EntityPtr entity, MaterialPtr rootMat, const std::string& directory) {
        // set the transformation info
        auto mat = node->mTransformation;
        aiVector3t<float> scale;
        aiQuaterniont<float> quat;
        aiVector3t<float> position;
        mat.Decompose(scale, quat, position);

        auto rotation = quat.GetMatrix();
        // @see https://stackoverflow.com/questions/15022630/how-to-calculate-the-angle-from-rotation-matrix
        const Radians angleX = Radians(std::atan2f(rotation.c2, rotation.c3));
        const Radians angleY = Radians(std::atan2f(-rotation.c1, std::sqrtf(rotation.c2 * rotation.c2 + rotation.c3 * rotation.c3)));
        const Radians angleZ = Radians(std::atan2f(rotation.b1, rotation.a1));

        entity->SetLocalPosRotScale(glm::vec3(position.x, position.y, position.z), Rotation(angleX, angleY, angleZ), glm::vec3(scale.x, scale.y, scale.z));
        RenderNodePtr rnode = RenderNodePtr(new RenderNode());
        entity->SetRenderNode(rnode);

        // Process all node meshes (if any)
        for (uint32_t i = 0; i < node->mNumMeshes; ++i) {
            aiMesh * mesh = scene->mMeshes[node->mMeshes[i]];
            ProcessMesh(rnode, mesh, scene, rootMat, directory);
        }

        // Now do the same for each child
        for (uint32_t i = 0; i < node->mNumChildren; ++i) {
            // Create a new container entity
            EntityPtr centity = Entity::Create();
            entity->AttachChild(centity);
            ProcessNode(node->mChildren[i], scene, centity, rootMat, directory);
        }
    }

    EntityPtr ResourceManager::_LoadModel(const std::string& name) {
        STRATUS_LOG << "Attempting to load model: " << name << std::endl;

        Assimp::Importer importer;
        // importer.SetPropertyInteger(AI_CONFIG_PP_SLM_VERTEX_LIMIT, 65000);
        // importer.SetPropertyInteger(AI_CONFIG_PP_SLM_TRIANGLE_LIMIT, 65000);
        //const aiScene *scene = importer.ReadFile(filename, aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_CalcTangentSpace | aiProcess_GenNormals | aiProcess_GenUVCoords);
        //const aiScene *scene = importer.ReadFile(filename, aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_FlipUVs | aiProcess_CalcTangentSpace | aiProcess_OptimizeMeshes);
        const aiScene *scene = importer.ReadFile(name, aiProcess_Triangulate | 
                                                       aiProcess_JoinIdenticalVertices |
                                                       aiProcess_SortByPType |
                                                       aiProcess_GenNormals |
                                                    //    aiProcess_GenSmoothNormals | 
                                                       aiProcess_FlipUVs | 
                                                       aiProcess_GenUVCoords | 
                                                       aiProcess_CalcTangentSpace |
                                                       aiProcess_SplitLargeMeshes | 
                                                       aiProcess_ImproveCacheLocality |
                                                       aiProcess_OptimizeMeshes |
                                                       aiProcess_OptimizeGraph |
                                                       aiProcess_FixInfacingNormals
                                                );

        auto material = MaterialManager::Instance()->CreateMaterial(name);

        if(!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
            STRATUS_ERROR << "Error loading model: " << name << std::endl << importer.GetErrorString() << std::endl;
            return nullptr;
        }

        EntityPtr e = Entity::Create();
        const std::string directory = name.substr(0, name.find_last_of('/'));
        ProcessNode(scene->mRootNode, scene, e, material, directory);

        auto ul = _LockWrite();
        // Create an internal copy for thread safety
        _loadedModels.insert(std::make_pair(name, Async<Entity>(e->Copy())));

        STRATUS_LOG << "Model loaded [" << name << "]" << std::endl;

        return e;
    }

    std::shared_ptr<ResourceManager::RawTextureData> ResourceManager::_LoadTexture(const std::string& file, const TextureHandle handle) {
        STRATUS_LOG << "Attempting to load texture from file: " << file << std::endl;
        std::shared_ptr<RawTextureData> texdata;
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

            texdata = std::make_shared<RawTextureData>();
            texdata->config = config;
            texdata->handle = handle;
            texdata->sizeBytes = width * height * numChannels * sizeof(uint8_t);
            texdata->data = data;
        } 
        else {
            STRATUS_ERROR << "Could not load texture: " << file << std::endl;
            return nullptr;
        }
    
        // auto ul = _LockWrite();
        // _loadedTextures.insert(std::make_pair(handle, Async<Texture>(*Engine::Instance()->GetMainThread(), [this, texdata]() {
        //     return _FinalizeTexture(*texdata);
        // })));
        return texdata;
    }

    Texture * ResourceManager::_FinalizeTexture(const RawTextureData& data) {
        Texture * texture = new Texture(data.config, data.data, false);
        texture->_setHandle(data.handle);
        texture->setCoordinateWrapping(TextureCoordinateWrapping::REPEAT);
        texture->setMinMagFilter(TextureMinificationFilter::LINEAR_MIPMAP_LINEAR, TextureMagnificationFilter::LINEAR);
        stbi_image_free(data.data);
        return texture;
    }

    EntityPtr ResourceManager::CreateCube() {
        return _cube->Copy();
    }

    EntityPtr ResourceManager::CreateQuad() {
        return _quad->Copy();
    }

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

    static const std::vector<GLfloat> quadData = std::vector<GLfloat>{
        // positions            normals                 texture coordinates     // tangents  // bitangents
        -1.0f, -1.0f, 0.0f,     0.0f, 0.0f, -1.0f,		0.0f, 0.0f,             1, 0, 0,     0, 1, 0,
        1.0f, -1.0f, 0.0f,     0.0f, 0.0f, -1.0f,		1.0f, 0.0f,             1, 0, 0,     0, 1, 0,
        1.0f,  1.0f, 0.0f,	    0.0f, 0.0f, -1.0f,		1.0f, 1.0f,             1, 0, 0,     0, 1, 0,
        -1.0f, -1.0f, 0.0f,     0.0f, 0.0f, -1.0f,      0.0f, 0.0f,             1, 0, 0,     0, 1, 0,
        1.0f,  1.0f, 0.0f,	    0.0f, 0.0f, -1.0f,		1.0f, 1.0f,             1, 0, 0,     0, 1, 0,
        -1.0f,  1.0f, 0.0f,	    0.0f, 0.0f, -1.0f,		0.0f, 1.0f,             1, 0, 0,     0, 1, 0,
    };

    void ResourceManager::_InitCube() {
        _cube = Entity::Create();
        RenderNodePtr rnode(new RenderNode());
        RenderMeshPtr rmesh(new RenderMesh());
        MaterialPtr mat = MaterialManager::Instance()->CreateDefault();

        const size_t cubeStride = 14;
        const size_t cubeNumVertices = cubeData.size() / cubeStride;

        for (size_t i = 0, f = 0; i < cubeNumVertices; ++i, f += cubeStride) {
            rmesh->AddVertex(glm::vec3(cubeData[f], cubeData[f + 1], cubeData[f + 2]));
            rmesh->AddNormal(glm::vec3(cubeData[f + 3], cubeData[f + 4], cubeData[f + 5]));
            rmesh->AddUV(glm::vec2(cubeData[f + 6], cubeData[f + 7]));
            // rmesh->AddTangent(glm::vec3(cubeData[f + 8], cubeData[f + 9], cubeData[f + 10]));
            // rmesh->AddBitangent(glm::vec3(cubeData[f + 11], cubeData[f + 12], cubeData[f + 13]));
        }

        rmesh->GenerateCpuData();
        rnode->AddMeshContainer(RenderMeshContainer{rmesh, mat});
        rnode->SetFaceCullMode(RenderFaceCulling::CULLING_CCW);
        _pendingFinalize.insert(std::make_pair("DefaultCube", Async<Entity>(_cube)));
        _cube->SetRenderNode(rnode);
    }

    void ResourceManager::_InitQuad() {
        _quad = Entity::Create();
        RenderNodePtr rnode(new RenderNode());
        RenderMeshPtr rmesh(new RenderMesh());
        MaterialPtr mat = MaterialManager::Instance()->CreateDefault();

        const size_t quadStride = 14;
        const size_t quadNumVertices = quadData.size() / quadStride;

        for (size_t i = 0, f = 0; i < quadNumVertices; ++i, f += quadStride) {
            rmesh->AddVertex(glm::vec3(quadData[f], quadData[f + 1], quadData[f + 2]));
            rmesh->AddNormal(glm::vec3(quadData[f + 3], quadData[f + 4], quadData[f + 5]));
            rmesh->AddUV(glm::vec2(quadData[f + 6], quadData[f + 7]));
            // rmesh->AddTangent(glm::vec3(quadData[f + 8], quadData[f + 9], quadData[f + 10]));
            // rmesh->AddBitangent(glm::vec3(quadData[f + 11], quadData[f + 12], quadData[f + 13]));
        }

        rmesh->GenerateCpuData();
        rnode->AddMeshContainer(RenderMeshContainer{rmesh, mat});
        rnode->SetFaceCullMode(RenderFaceCulling::CULLING_NONE);
        _pendingFinalize.insert(std::make_pair("DefaultQuad", Async<Entity>(_quad)));
        _quad->SetRenderNode(rnode);
    }
}