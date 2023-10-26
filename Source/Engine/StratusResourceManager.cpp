#include "StratusResourceManager.h"
#include "StratusEngine.h"
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/pbrmaterial.h>
#include <assimp/material.h>
#include <assimp/GltfMaterial.h>
#include "StratusRendererFrontend.h"
#include "StratusApplicationThread.h"
#include "StratusTaskSystem.h"
#include "StratusAsync.h"
#include "StratusRenderComponents.h"
#include "StratusTransformComponent.h"
#include <sstream>
#include <algorithm>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "meshoptimizer.h"

namespace stratus {
    ResourceManager::ResourceManager() {}

    ResourceManager::~ResourceManager() {
    }

    SystemStatus ResourceManager::Update(const double deltaSeconds) {
        {
            auto ul = LockWrite_();
            ClearAsyncTextureData_();
            ClearAsyncModelData_();
        }

        return SystemStatus::SYSTEM_CONTINUE;
    }

    bool ResourceManager::Initialize() {
        InitCube_();
        InitQuad_();

        return true;
    }

    void ResourceManager::Shutdown() {
        loadedModels_.clear();
        pendingFinalize_.clear();
        meshFinalizeQueue_.clear();
        asyncLoadedTextureData_.clear();
        loadedTextures_.clear();
        loadedTexturesByFile_.clear();
    }

    void ResourceManager::ClearAsyncTextureData_() {
        std::vector<TextureHandle> toDelete;
        //constexpr usize maxBytes = 1024 * 1024 * 10; // 10 mb per frame
        usize totalTex = 0;
        usize totalBytes = 0;
        std::vector<TextureHandle> handles;
        std::vector<std::shared_ptr<RawTextureData>> rawTexData;
        for (auto& tpair : asyncLoadedTextureData_) {
            //if (totalBytes > maxBytes) break;
            if (tpair.second.Completed()) {
                toDelete.push_back(tpair.first);
                if (tpair.second.Failed()) {
                    continue;
                }
                TextureHandle handle = tpair.first;
                auto texdata = tpair.second.GetPtr();
                if (!texdata) continue;
                totalBytes += texdata->sizeBytes;
                ++totalTex;

                handles.push_back(handle);
                rawTexData.push_back(texdata);
            }
        }

        if (totalBytes > 0) {
            STRATUS_LOG << "Texture data bytes processed: " << totalBytes << ", " << totalTex << std::endl;
        }

        for (auto handle : toDelete) asyncLoadedTextureData_.erase(handle);

        ApplicationThread::Instance()->Queue([this, handles, rawTexData]() {
            std::vector<Texture *> ptrs(rawTexData.size());
            for (i32 i = 0; i < rawTexData.size(); ++i) {
                ptrs[i] = FinalizeTexture_(*rawTexData[i]);
            }

            auto ul = LockWrite_();
            for (i32 i = 0; i < handles.size(); ++i) {
                texturesStillLoading_.erase(handles[i]);
                loadedTextures_.insert(std::make_pair(handles[i], Async<Texture>(std::shared_ptr<Texture>(ptrs[i]))));
            }
        });
    }

    void ResourceManager::ClearAsyncModelData_() {
        static constexpr usize maxModelBytesPerFrame = 1024 * 1024 * 2;
        // First generate GPU data for some of the meshes
        usize totalBytes = 0;
        std::vector<MeshletPtr> removeFromGpuDataQueue;
        for (auto meshlet : generateMeshGpuDataQueue_) {
            meshlet->FinalizeData();
            totalBytes += meshlet->GetGpuSizeBytes();
            removeFromGpuDataQueue.push_back(meshlet);
            if (removeFromGpuDataQueue.size() > 30 || totalBytes >= maxModelBytesPerFrame) break;
            //if (totalBytes >= maxModelBytesPerFrame) break;
        }

        for (auto mesh : removeFromGpuDataQueue) generateMeshGpuDataQueue_.erase(mesh);

        if (totalBytes > 0) STRATUS_LOG << "Processed " << totalBytes << " bytes of mesh data: " << removeFromGpuDataQueue.size() << " meshes" << std::endl;

        // If none other left to finalize, end early
        if (pendingFinalize_.size() == 0) return;

        //constexpr usize maxBytes = 1024 * 1024 * 10; // 10 mb per frame
        std::vector<std::string> toDelete;
        for (auto& mpair : pendingFinalize_) {
            if (mpair.second.Completed() && !mpair.second.Failed()) {
                toDelete.push_back(mpair.first);
                auto ptr = mpair.second.GetPtr();

                ClearAsyncModelData_(ptr);
           }
        }

        for (const std::string& name : toDelete) {
           pendingFinalize_.erase(name);
        }

        if (meshFinalizeQueue_.size() == 0) return;

        //usize totalBytes = 0;
        std::vector<MeshletPtr> meshesToDelete(meshFinalizeQueue_.size());
        usize idx = 0;
        for (auto& meshlet : meshFinalizeQueue_) {
            meshesToDelete[idx] = meshlet;
            ++idx;
        }

        meshFinalizeQueue_.clear();

        // for (MeshPtr mesh : _meshFinalizeQueue) {
        //     mesh->FinalizeData();
        //     totalBytes += mesh->GetGpuSizeBytes();
        //     meshesToDelete.push_back(mesh);
        //     //if (totalBytes >= maxBytes) break;
        // }

        std::vector<Async<bool>> waiting;
        const usize numThreads = INSTANCE(TaskSystem)->Size();
        for (usize i = 0; i < numThreads; ++i) {
            const usize threadIndex = i;

            const auto process = [this, threadIndex, numThreads, meshesToDelete]() {
                STRATUS_LOG << "Processing " << meshesToDelete.size() << " as a task group" << std::endl;
                for (usize i = threadIndex; i < meshesToDelete.size(); i += numThreads) {
                    MeshletPtr mesh = meshesToDelete[i];
                    mesh->PackCpuData();
                    mesh->CalculateAabbs(glm::mat4(1.0f));
                    mesh->GenerateLODs();
                }

                return new bool(true);
            };

           waiting.push_back(INSTANCE(TaskSystem)->ScheduleTask<bool>(process));
        }

        const auto callback = [this, meshesToDelete](auto) {
            for (auto mesh : meshesToDelete) {
                generateMeshGpuDataQueue_.insert(mesh);
            }
        };

        INSTANCE(TaskSystem)->AddTaskGroupCallback<bool>(callback, waiting);

        // for (auto& wait : waiting) {
        //    while (!wait.Completed())
        //        ;
        // }

        // for (auto mesh : meshesToDelete) {
        //    mesh->FinalizeData();
        //    totalBytes += mesh->GetGpuSizeBytes();
        // }

        // for (MeshPtr mesh : meshesToDelete) _meshFinalizeQueue.erase(mesh);

        //if (totalBytes > 0) STRATUS_LOG << "Processed " << totalBytes << " bytes of mesh data: " << meshesToDelete.size() << " meshes" << std::endl;
    }

    void ResourceManager::ClearAsyncModelData_(EntityPtr ptr) {
        if (ptr == nullptr) return;
        for (auto& child : ptr->GetChildNodes()) {
            ClearAsyncModelData_(child);
        }

        auto rnode = ptr->Components().GetComponent<RenderComponent>().component;
        if (rnode == nullptr) return;

        for (i32 i = 0; i < rnode->meshes->meshes.size(); ++i) {
            auto mesh = rnode->meshes->meshes[i];
            for (i32 j = 0; j < mesh->NumMeshlets(); ++j) {
                meshFinalizeQueue_.insert(mesh->GetMeshlet(j));
            }
        }
    }

    Async<Entity> ResourceManager::LoadModel(const std::string& name, const ColorSpace& cspace, const bool optimizeGraph, RenderFaceCulling defaultCullMode) {
        {
            auto sl = LockRead_();
            if (loadedModels_.find(name) != loadedModels_.end()) {
                Async<Entity> e = loadedModels_.find(name)->second;
                return (e.Completed() && !e.Failed()) ? Async<Entity>(e.GetPtr()->Copy()) : e;
            }
        }

        auto ul = LockWrite_();
        TaskSystem * tasks = TaskSystem::Instance();
        Async<Entity> e = tasks->ScheduleTask<Entity>([this, name, defaultCullMode, optimizeGraph, cspace]() {
            return LoadModel_(name, cspace, optimizeGraph, defaultCullMode);
        });

        loadedModels_.insert(std::make_pair(name, e));
        pendingFinalize_.insert(std::make_pair(name, e));
        return e;
    }

    TextureHandle ResourceManager::LoadTexture(const std::string& name, const ColorSpace& cspace) {
        return LoadTextureImpl_({ name }, {}, cspace);
    }

    TextureHandle ResourceManager::LoadTexture(const std::string& name, BinaryDataWrapper data, const ColorSpace& cspace) {
        return LoadTextureImpl_({ name }, { data }, cspace);
    }

    TextureHandle ResourceManager::LoadCubeMap(const std::string& prefix, const ColorSpace& cspace, const std::string& fileExt) {
        return LoadTextureImpl_({prefix + "right." + fileExt,
                                 prefix + "left." + fileExt,
                                 prefix + "top." + fileExt,
                                 prefix + "bottom." + fileExt,
                                 prefix + "front." + fileExt,
                                 prefix + "back." + fileExt}, 
                                {},
                                cspace,
                                TextureType::TEXTURE_CUBE_MAP,
                                TextureCoordinateWrapping::CLAMP_TO_EDGE,
                                TextureMinificationFilter::LINEAR_MIPMAP_LINEAR,
                                TextureMagnificationFilter::LINEAR);
    }

    TextureHandle ResourceManager::LoadTextureImpl_(const std::vector<std::string>& files, 
                                                    const std::vector<BinaryDataWrapper>& data,
                                                    const ColorSpace& cspace,
                                                    const TextureType type,
                                                    const TextureCoordinateWrapping wrap,
                                                    const TextureMinificationFilter min,
                                                    const TextureMagnificationFilter mag) {
        if (files.size() == 0) return TextureHandle::Null();

        // Generate a lookup name by combining all texture files into a single string
        std::stringstream lookup;
        for (const std::string& file : files) lookup << file;
        const std::string name = lookup.str();

        {
            // Check if we have already loaded this texture file combination before
            auto sl = LockRead_();
            if (loadedTexturesByFile_.find(name) != loadedTexturesByFile_.end()) {
                return loadedTexturesByFile_.find(name)->second;
            }
        }

        auto ul = LockWrite_();
        auto handle = TextureHandle::NextHandle();
        TaskSystem * tasks = TaskSystem::Instance();
        // We have to use the main thread since Texture calls glGenTextures :(
        Async<RawTextureData> as = tasks->ScheduleTask<RawTextureData>([this, files, data, handle, cspace, type, wrap, min, mag]() {
            auto result = LoadTexture_(files, data, handle, cspace, type, wrap, min, mag);
            //auto ul = this->LockWrite_();
            //this->texturesStillLoading_.erase(handle);
            return result;
        });

        texturesStillLoading_.insert(handle);
        loadedTexturesByFile_.insert(std::make_pair(name, handle));
        asyncLoadedTextureData_.insert(std::make_pair(handle, as));

        return handle;
    }

    Texture ResourceManager::LookupTexture(const TextureHandle handle, TextureLoadingStatus& status) const {
        auto sl = LockRead_();
        if (loadedTextures_.find(handle) == loadedTextures_.end()) {
            if (texturesStillLoading_.find(handle) == texturesStillLoading_.end()) {
                status = TextureLoadingStatus::FAILED;
            }
            else {
                status = TextureLoadingStatus::LOADING;
            }

            return Texture();
        }

        status = TextureLoadingStatus::LOADING_DONE;

        return loadedTextures_.find(handle)->second.Get();
    }

    static TextureHandle LoadMaterialTexture(const aiScene * scene, aiMaterial * mat, const aiTextureType& type, const std::string& directory, const ColorSpace& cspace) {
        TextureHandle texture;
        if (mat->GetTextureCount(type) > 0) {
            aiString str; 
            mat->GetTexture(type, 0, &str);
            std::string file = str.C_Str();

            const auto last = file.find_last_of('*');
            if (last != file.npos) {
                const auto embeddedIndex = std::stoi(file.substr(last + 1));
                const auto embeddedTexture = scene->mTextures[embeddedIndex];

                if (embeddedTexture->mHeight != 0) {
                    STRATUS_ERROR << "Non-compressed embedded texture found - skipping" << std::endl;
                    return TextureHandle::Null();
                }

                const usize sizeBytes = embeddedTexture->mWidth;
                u8 * writeData = new u8[sizeBytes];
                const u8 * readData = reinterpret_cast<const u8 *>(embeddedTexture->pcData);

                std::memcpy(writeData, readData, sizeBytes);

                BinaryDataWrapper binaryData;
                binaryData.data = writeData;
                binaryData.sizeBytes = 4 * sizeBytes;

                texture = ResourceManager::Instance()->LoadTexture(directory + "/" + file, binaryData, cspace);
            }
            else {
                texture = ResourceManager::Instance()->LoadTexture(directory + "/" + file, cspace);
            }
        }

        return texture;
    }

    static void PrintMatType(const aiMaterial * aimat, const aiTextureType type) {
        static const std::unordered_map<i32, std::string> conversion = {
            {aiTextureType_DIFFUSE, "Diffuse"},
            {aiTextureType_SPECULAR, "Specular"},
            {aiTextureType_AMBIENT, "Ambient"},
            {aiTextureType_EMISSIVE, "Emissive"},
            {aiTextureType_OPACITY, "Opacity"},
            {aiTextureType_HEIGHT, "Height"},
            {aiTextureType_NORMALS, "Normals"},
            {aiTextureType_BASE_COLOR, "Base Color"},
            //{aiTextureType_NORMAL_CAMERA, "Normal Camera"},
            {aiTextureType_EMISSION_COLOR, "Emission Color"},
            {aiTextureType_METALNESS, "Metalness"},
            //{aiTextureType_AMBIENT_OCCLUSION, "Ambient Occlusion"},
            //{aiTextureType_DIFFUSE_ROUGHNESS, "Diffuse Roughness"},
            //{aiTextureType_SHEEN, "Sheen"},
            {aiTextureType_CLEARCOAT, "Clearcoat"},
            {aiTextureType_TRANSMISSION, "Transmission"},
            {aiTextureType_UNKNOWN, "Unknown/Other"}
        };

        if (conversion.find(type) == conversion.end()) return;

        const auto count = aimat->GetTextureCount(type);
        std::stringstream out;
        out << count;
        if (count > 0) {
            aiString str;
            aimat->GetTexture(type, 0, &str);
            std::string file = str.C_Str();
            out << ": " << conversion.find(type)->second << ", " << file;
        }
        out << std::endl;
        STRATUS_LOG << out.str();
    }

    struct MeshToProcess_ {
        aiMesh * aim;
        MeshPtr mesh;
        MaterialPtr material;
    };

    //static void ProcessMesh(
    //    RenderComponent * renderNode, 
    //    const aiMatrix4x4& transform, 
    //    aiMesh * mesh, 
    //    const aiScene * scene, 
    //    MaterialPtr rootMat, 
    //    const std::string& directory, 
    //    const std::string& extension, 
    //    RenderFaceCulling defaultCullMode, 
    //    const ColorSpace& cspace) {
    static void ProcessMesh(
        MeshToProcess_& processMesh,
        const aiScene * scene, 
        const std::string& directory, 
        const std::string& extension, 
        RenderFaceCulling defaultCullMode, 
        const ColorSpace& cspace) {
        
        //if (mesh->mNumUVComponents[0] == 0) return;
        //if (mesh->mNormals == nullptr || mesh->mTangents == nullptr || mesh->mBitangents == nullptr) return;
        //if (mesh->mNormals == nullptr) return;

        //MeshPtr rmesh = Mesh::Create();
        aiMesh * mesh = processMesh.aim;
        MeshPtr rmesh = processMesh.mesh;
        //auto meshlet = rmesh->NewMeshlet();

        // Split mesh into meshlets
        const size_t maxVertices = 64;
        const size_t maxTriangles = 124;
        const float coneWeight = 1.0f;

        std::vector<u32> indices(mesh->mNumFaces * 3);
        for (u32 i = 0; i < mesh->mNumFaces; i++) {
            for (u32 index = 0; index < 3; ++index) {
                indices[3 * i + index] = mesh->mFaces[i].mIndices[index];
            }
            //aiFace face = mesh->mFaces[i];
            //for (u32 j = 0; j < face.mNumIndices; j++) {
            //    indices.push_back(face.mIndices[j]);
            //}
        }

        //std::vector<float> vertexData(3 * mesh->mNumVertices);
        //for (usize i = 0; i < mesh->mNumVertices; ++i) {
        //    vertexData[3 * i] = mesh->mVertices[i].x;
        //    vertexData[3 * i + 1] = mesh->mVertices[i].y;
        //    vertexData[3 * i + 2] = mesh->mVertices[i].z;
        //}

        // const size_t maxMeshlets = meshopt_buildMeshletsBound(indices.size(), maxVertices, maxTriangles);
        // std::vector<meshopt_Meshlet> meshlets(maxMeshlets);
        // std::vector<u32> meshletVertices(maxMeshlets * maxVertices);
        // std::vector<u8> meshletTriangles(maxMeshlets * maxTriangles * 3);

        // size_t meshletCount = meshopt_buildMeshlets(
        //     meshlets.data(), 
        //     meshletVertices.data(), 
        //     meshletTriangles.data(), 
        //     indices.data(),
        //     indices.size(),
        //     reinterpret_cast<float *>(mesh->mVertices),
        //     mesh->mNumVertices,
        //     3,
        //     maxVertices, 
        //     maxTriangles, 
        //     coneWeight
        // );

        // Trim data
        //const meshopt_Meshlet& last = meshlets[meshletCount - 1];

        //meshletVertices.resize(last.vertex_offset + last.vertex_count);
        //meshletTriangles.resize(last.triangle_offset + ((last.triangle_count * 3 + 3) & ~3));
        //meshlets.resize(meshletCount);

        //STRATUS_LOG << "MESHLETS: " << meshletCount << std::endl;

        for (usize i = 0; i < 1; ++i) {
        auto meshlet = rmesh->NewMeshlet();

        meshlet->ReserveVertices(mesh->mNumVertices);
        meshlet->ReserveIndices(indices.size());

        // Process core primitive data
        // TODO: Remove duplicate data from meshlets!
        for (u32 mi = 0; mi < mesh->mNumVertices; mi++) {
            const auto index = mi; //meshletVertices[meshoptMeshlet.vertex_offset + mi];
            meshlet->AddVertex(glm::vec3(mesh->mVertices[index].x, mesh->mVertices[index].y, mesh->mVertices[index].z));
            meshlet->AddNormal(glm::vec3(mesh->mNormals[index].x, mesh->mNormals[index].y, mesh->mNormals[index].z));

            if (mesh->mNumUVComponents[0] != 0) {
                meshlet->AddUV(glm::vec2(mesh->mTextureCoords[0][index].x, mesh->mTextureCoords[0][index].y));
            }
            else {
                meshlet->AddUV(glm::vec2(1.0f, 1.0f));
            }

            if (mesh->mTangents != nullptr)   meshlet->AddTangent(glm::vec3(mesh->mTangents[index].x, mesh->mTangents[index].y, mesh->mTangents[index].z));
            if (mesh->mBitangents != nullptr) meshlet->AddBitangent(glm::vec3(mesh->mBitangents[index].x, mesh->mBitangents[index].y, mesh->mBitangents[index].z));
        }

        // Process indices
        for (u32 i = 0; i < mesh->mNumFaces; i++) {
           aiFace face = mesh->mFaces[i];
           for (u32 j = 0; j < face.mNumIndices; j++) {
               meshlet->AddIndex(face.mIndices[j]);
           }
        }
        }

        //for (usize meshIndex = 0; meshIndex < meshletCount; ++meshIndex) {
        //    const meshopt_Meshlet& meshoptMeshlet = meshlets[meshIndex];
        //    if (meshoptMeshlet.vertex_count == 0) {
        //        continue;
        //    }

        //    auto meshlet = rmesh->NewMeshlet();

        //    meshlet->ReserveVertices(meshoptMeshlet.vertex_count);
        //    meshlet->ReserveIndices(meshoptMeshlet.vertex_count);
        //    //meshlet->ReserveVertices(mesh->mNumVertices);
        //    //meshlet->ReserveIndices(indices.size());

        //    // Process core primitive data
        //    // TODO: Remove duplicate data from meshlets!
        //    for (u32 mi = 0; mi < meshoptMeshlet.vertex_count; mi++) {
        //    //for (u32 mi = 0; mi < mesh->mNumVertices; mi++) {
        //        const auto index = meshletVertices[meshoptMeshlet.vertex_offset + mi];
        //        meshlet->AddVertex(glm::vec3(mesh->mVertices[index].x, mesh->mVertices[index].y, mesh->mVertices[index].z));
        //        meshlet->AddNormal(glm::vec3(mesh->mNormals[index].x, mesh->mNormals[index].y, mesh->mNormals[index].z));
        //        meshlet->AddIndex(mi);

        //        if (mesh->mNumUVComponents[0] != 0) {
        //            meshlet->AddUV(glm::vec2(mesh->mTextureCoords[0][index].x, mesh->mTextureCoords[0][index].y));
        //        }
        //        else {
        //            meshlet->AddUV(glm::vec2(1.0f, 1.0f));
        //        }

        //        if (mesh->mTangents != nullptr)   meshlet->AddTangent(glm::vec3(mesh->mTangents[index].x, mesh->mTangents[index].y, mesh->mTangents[index].z));
        //        if (mesh->mBitangents != nullptr) meshlet->AddBitangent(glm::vec3(mesh->mBitangents[index].x, mesh->mBitangents[index].y, mesh->mBitangents[index].z));
        //    }

        //     //for (u32 mi = 0; mi < meshoptMeshlet.vertex_count; mi++) {
        //     //    const auto index = meshletVertices[meshoptMeshlet.vertex_offset + mi];
        //     //    meshlet->AddIndex(index);
        //     //}
        //}

        // Process material
        //MaterialPtr m = rootMat->CreateSubMaterial();
        MaterialPtr m = processMesh.material;

        RenderFaceCulling cull = defaultCullMode;
        if (mesh->mMaterialIndex >= 0) {
            aiMaterial * aimat = scene->mMaterials[mesh->mMaterialIndex];

            // Disable face culling if applicable
            i32 doubleSided = 0;
            if(AI_SUCCESS == aimat->Get(AI_MATKEY_TWOSIDED, doubleSided)) {
                //STRATUS_LOG << "Double Sided: " << doubleSided << std::endl;
                if (doubleSided != 0) {
                    cull = RenderFaceCulling::CULLING_NONE;
                }
            }
        }

        // const glm::mat4 gt = ToMat4(transform);
        // renderNode->meshes->meshes.push_back(rmesh);
        // renderNode->meshes->transforms.push_back(gt);
        // renderNode->AddMaterial(m);
        rmesh->SetFaceCulling(cull);
    }

    static void ProcessMaterial(
        const aiScene* scene,
        const aiMesh* mesh,
        MaterialPtr material,
        const std::string& directory,
        const std::string& extension,
        const ColorSpace& cspace) {

        if (mesh->mMaterialIndex >= 0) {
            aiMaterial* aimat = scene->mMaterials[mesh->mMaterialIndex];

            aiString matName;
            aimat->Get<aiString>(AI_MATKEY_NAME, matName);
            STRATUS_LOG << "Loading Mesh Material [" << material->GetName() << "]" << std::endl;
            // PrintMatType(aimat, aiTextureType_DIFFUSE);
            // PrintMatType(aimat, aiTextureType_SPECULAR);
            // PrintMatType(aimat, aiTextureType_AMBIENT);
            // PrintMatType(aimat, aiTextureType_EMISSIVE);
            // PrintMatType(aimat, aiTextureType_HEIGHT);
            // PrintMatType(aimat, aiTextureType_NORMALS);
            // PrintMatType(aimat, aiTextureType_OPACITY);
            // PrintMatType(aimat, aiTextureType_BASE_COLOR);
            // PrintMatType(aimat, aiTextureType_NORMAL_CAMERA);
            // PrintMatType(aimat, aiTextureType_EMISSION_COLOR);
            // PrintMatType(aimat, aiTextureType_METALNESS);
            // PrintMatType(aimat, aiTextureType_AMBIENT_OCCLUSION);
            // PrintMatType(aimat, aiTextureType_DIFFUSE_ROUGHNESS);
            // PrintMatType(aimat, aiTextureType_SHEEN);
            // PrintMatType(aimat, aiTextureType_CLEARCOAT);
            // PrintMatType(aimat, aiTextureType_TRANSMISSION);
            // PrintMatType(aimat, aiTextureType_UNKNOWN);

            aiColor4D diffuse;
            aiColor4D reflective;
            aiColor4D specular;
            aiColor4D emissive;
            f32 metallic;
            f32 roughness;
            f32 opacity;
            f32 specularFactor;
            u32 max = 1;
            aiColor4D transparency;

            if (aiGetMaterialFloat(aimat, AI_MATKEY_METALLIC_FACTOR, &metallic) == AI_SUCCESS) {
                material->SetMetallic(metallic);
                //STRATUS_LOG << "M: " << metallic << std::endl;
            }
            if (aiGetMaterialFloat(aimat, AI_MATKEY_ROUGHNESS_FACTOR, &roughness) == AI_SUCCESS) {
                material->SetRoughness(roughness);
            }

            if (aiGetMaterialColor(aimat, AI_MATKEY_COLOR_DIFFUSE, &diffuse) == AI_SUCCESS) {
                material->SetDiffuseColor(glm::vec4(diffuse.r, diffuse.g, diffuse.b, std::clamp(diffuse.a, 0.0f, 1.0f)));
            }
            if (aiGetMaterialColor(aimat, AI_MATKEY_COLOR_EMISSIVE, &emissive) == AI_SUCCESS) {
                material->SetEmissiveColor(glm::vec3(emissive.r, emissive.g, emissive.b));
            }
            else {
                material->SetEmissiveColor(glm::vec3(0.0f));
            }
            if (aiGetMaterialColor(aimat, AI_MATKEY_COLOR_REFLECTIVE, &reflective) == AI_SUCCESS) {
                material->SetReflectance(std::max<f32>(reflective.r, std::max<f32>(reflective.g, reflective.b)));
                //material->SetBaseReflectivity(glm::vec3(reflective.r, reflective.g, reflective.b));
                //material->SetMaxReflectivity(glm::vec3(reflective.r, reflective.g, reflective.b));
                //STRATUS_LOG << "RF: " << reflective.r << " " << reflective.g << " " << reflective.b << std::endl;
            }
            //else if (aiGetMaterialColor(aimat, AI_MATKEY_BASE_COLOR, &reflective) == AI_SUCCESS) {
            //    m->SetBaseReflectivity(glm::vec3(reflective.r, reflective.g, reflective.b));
            //    STRATUS_LOG << "RF: " << reflective.r << " " << reflective.g << " " << reflective.b << std::endl;
            //}
            else if (aiGetMaterialFloatArray(aimat, AI_MATKEY_REFRACTI, &specularFactor, &max) == AI_SUCCESS) {
                //STRATUS_LOG << "SP: " << specularFactor << " " << max << std::endl;
                f32 reflectance = (specularFactor - 1.0) / (specularFactor + 1.0);
                reflectance = reflectance * reflectance;
                material->SetReflectance(reflectance);
                //material->SetBaseReflectivity(glm::vec3(reflectance));
                //material->SetMaxReflectivity(glm::vec3(reflectance));
                //STRATUS_LOG << "Reflectance: " << reflectance << std::endl;
            }
            if (aiGetMaterialFloat(aimat, AI_MATKEY_GLOSSINESS_FACTOR, &specularFactor) == AI_SUCCESS) {
                // STRATUS_LOG << "G: " << specularFactor << std::endl;
            }

            // STRATUS_LOG << "RMS: " << roughness << " " << metallic << " " << specularFactor << " " << diffuse.r << " " << diffuse.g << " " << diffuse.b << std::endl;

            //     STRATUS_LOG << "Opacity Value: " << opacity << std::endl;
            // }
            // if (aiGetMaterialColor(aimat, AI_MATKEY_COLOR_TRANSPARENT, &transparency) == AI_SUCCESS) {
            //     STRATUS_LOG << "Transparency: " << transparency.r << ", " << transparency.g << ", " << transparency.b << ", " << transparency.a << std::endl;
            // }

            material->SetDiffuseMap(LoadMaterialTexture(scene, aimat, aiTextureType_DIFFUSE, directory, cspace));
            // Important: Unless the normal/depth maps were generated as sRGB textures, srgb must be set to false!
            auto normalMap = LoadMaterialTexture(scene, aimat, aiTextureType_NORMALS, directory, ColorSpace::NONE);
            if (normalMap != TextureHandle::Null()) {
                material->SetNormalMap(normalMap);
            }
            //else {
                //m->SetNormalMap(LoadMaterialTexture(aimat, aiTextureType_HEIGHT, directory, ColorSpace::LINEAR));
            //}
            //m->SetDepthMap(LoadMaterialTexture(aimat, aiTextureType_HEIGHT, directory, ColorSpace::LINEAR));
            material->SetRoughnessMap(LoadMaterialTexture(scene, aimat, aiTextureType_DIFFUSE_ROUGHNESS, directory, ColorSpace::NONE));
            material->SetEmissiveMap(LoadMaterialTexture(scene, aimat, aiTextureType_EMISSIVE, directory, ColorSpace::NONE));
            material->SetMetallicMap(LoadMaterialTexture(scene, aimat, aiTextureType_METALNESS, directory, ColorSpace::NONE));
            // GLTF 2.0 have the metallic-roughness map specified as aiTextureType_UNKNOWN at the time of writing
            // TODO: See if other file types encode metallic-roughness in the same way
            if (extension == "gltf" || extension == "GLTF") {
                material->SetMetallicRoughnessMap(LoadMaterialTexture(scene, aimat, aiTextureType_UNKNOWN, directory, ColorSpace::NONE));
            }

            // STRATUS_LOG << "m " 
            //     << m->GetDiffuseMap() << " "
            //     << m->GetNormalMap() << " "
            //     << m->GetDepthMap() << " "
            //     << m->GetRoughnessMap() << " "
            //     << m->GetAmbientTexture() << " "
            //     << m->GetMetallicMap() << " "
            //     << m->GetMetallicRoughnessMap() << std::endl;
        }

    }


    static void ProcessNode(
        aiNode * node, 
        const aiScene * scene, 
        EntityPtr entity, 
        const aiMatrix4x4& parentTransform, 
        const std::string& name,
        const std::string& directory, 
        const std::string& extension, 
        RenderFaceCulling defaultCullMode, 
        const ColorSpace& cspace,
        std::vector<MeshToProcess_>& meshes) {

        // set the transformation info
        aiMatrix4x4 aiMatTransform = node->mTransformation;
        // See https://assimp-docs.readthedocs.io/en/v5.1.0/usage/use_the_lib.html
        // ASSIMP uses row-major convention
        auto transform = parentTransform * aiMatTransform;// * parentTransform;

        if (node->mNumMeshes > 0) {
            InitializeRenderEntity(entity);
            auto rnode = entity->Components().GetComponent<RenderComponent>().component;
            auto gt = ToMat4(transform);

            // Process all node meshes (if any)
            for (u32 i = 0; i < node->mNumMeshes; ++i) {
                aiMesh * mesh = scene->mMeshes[node->mMeshes[i]];
                // aiMaterial* aimat = scene->mMaterials[mesh->mMaterialIndex];
                // STRATUS_LOG << "Material Idx / Num Properties " << mesh->mMaterialIndex << " / " << aimat->mNumProperties << std::endl;
                // std::vector<f32> values(4);
                // for (i32 i = 0; i < aimat->mNumProperties; ++i) {
                //     STRATUS_LOG << aimat->mProperties[i]->mKey.C_Str() << " " << aimat->mProperties[i]->mDataLength << std::endl;
                // }
                // aiColor4D emissive;
                // f32 metallic;
                // f32 roughness;
                // u32 max = 4;
                // f32 refracti;
                // aiGetMaterialFloatArray(aimat, AI_MATKEY_SHININESS, values.data(), &max);
                // aiGetMaterialColor(aimat, AI_MATKEY_COLOR_EMISSIVE, &emissive);
                // aiGetMaterialFloat(aimat, AI_MATKEY_METALLIC_FACTOR, &metallic);
                // aiGetMaterialFloat(aimat, AI_MATKEY_ROUGHNESS_FACTOR, &roughness);

                // aiGetMaterialFloatArray(aimat, AI_MATKEY_REFRACTI, &refracti, &max);

                // STRATUS_LOG << "RMS: " << roughness << " " << metallic << " " << values[0] << " " << max << std::endl;
                // STRATUS_LOG << "E: " << emissive.r << " " << emissive.g << " " << emissive.b << std::endl;
                // STRATUS_LOG << "RF: " << refracti << std::endl;

                //if (mesh->mNormals == nullptr || mesh->mTangents == nullptr || mesh->mBitangents == nullptr) continue;
                if (mesh->mNormals == nullptr) continue;
                // Attempt to find degenerate meshes
                if (mesh->mNumFaces > 0) {
                    u32 numIndices = 0;
                    for(u32 i = 0; i < mesh->mNumFaces; i++) {
                        aiFace face = mesh->mFaces[i];
                        numIndices += face.mNumIndices;
                    }

                    if (numIndices % 3 != 0) continue;
                }
                else {
                    if (mesh->mNumVertices % 3 != 0) continue;
                }
                
                auto stratusMesh = Mesh::Create();
                
                const std::string materialName = name + "#" + std::to_string(mesh->mMaterialIndex);
                MaterialPtr m = INSTANCE(MaterialManager)->GetMaterial(materialName);
                ProcessMaterial(scene, mesh, m, directory, extension, cspace);

                rnode->meshes->meshes.push_back(stratusMesh);
                rnode->meshes->transforms.push_back(gt);
                rnode->AddMaterial(m);
                MeshToProcess_ meshToProcess;
                meshToProcess.aim = mesh;
                meshToProcess.mesh = stratusMesh;
                meshToProcess.material = m;
                meshes.push_back(meshToProcess);
                //ProcessMesh(rnode, transform, mesh, scene, rootMat, directory, extension, defaultCullMode, cspace);
            }
        }

        // Now do the same for each child
        for (u32 i = 0; i < node->mNumChildren; ++i) {
            // Create a new container Entity
            EntityPtr centity = CreateTransformEntity();
            entity->AttachChildNode(centity);
            ProcessNode(node->mChildren[i], scene, centity, transform, name, directory, extension, defaultCullMode, cspace, meshes);
        }
    }

    EntityPtr ResourceManager::LoadModel_(const std::string& name, const ColorSpace& cspace, const bool optimizeGraph, RenderFaceCulling defaultCullMode) {
        STRATUS_LOG << "Attempting to load model: " << name << std::endl;

        Assimp::Importer importer;
        //importer.SetPropertyInteger(AI_CONFIG_PP_SLM_VERTEX_LIMIT, 16000);
        importer.SetPropertyInteger(AI_CONFIG_PP_SLM_TRIANGLE_LIMIT, 512);

        //u32 pflags = aiProcess_JoinIdenticalVertices | aiProcess_FindInvalidData;

        u32 pflags = aiProcess_Triangulate |
           aiProcess_JoinIdenticalVertices |
           aiProcess_SortByPType |
           aiProcess_GenNormals |
           //aiProcess_ValidateDataStructure |
           aiProcess_RemoveRedundantMaterials |
           aiProcess_SortByPType |
           //aiProcess_GenSmoothNormals | 
           aiProcess_FlipUVs |
           aiProcess_GenUVCoords |
           aiProcess_CalcTangentSpace |
           aiProcess_SplitLargeMeshes |
           aiProcess_ImproveCacheLocality |
           aiProcess_OptimizeMeshes |
           //aiProcess_OptimizeGraph |
           //aiProcess_FixInfacingNormals |
           aiProcess_FindDegenerates |
           aiProcess_FindInvalidData |
           aiProcess_FindInstances;

        if (optimizeGraph) {
           pflags |= aiProcess_OptimizeGraph;
        }

        //const aiScene *scene = importer.ReadFile(filename, aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_CalcTangentSpace | aiProcess_GenNormals | aiProcess_GenUVCoords);
        //const aiScene *scene = importer.ReadFile(filename, aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_FlipUVs | aiProcess_CalcTangentSpace | aiProcess_OptimizeMeshes);
        const aiScene *scene = importer.ReadFile(name, pflags);

        if(!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
            STRATUS_ERROR << "Error loading model: " << name << std::endl << importer.GetErrorString() << std::endl;
            return nullptr;
        }

        STRATUS_LOG << "IMPORTED: " << scene->mNumMeshes << std::endl;

        // Create all scene materials
        for (u32 i = 0; i < scene->mNumMaterials; ++i) {
            const std::string materialName = name + "#" + std::to_string(i);
            auto material = INSTANCE(MaterialManager)->CreateMaterial(materialName);
        }

        EntityPtr e = CreateTransformEntity();
        std::vector<MeshToProcess_> meshes;
        const std::string extension = name.substr(name.find_last_of('.') + 1, name.size());
        const std::string directory = name.substr(0, name.find_last_of('/'));
        ProcessNode(scene->mRootNode, scene, e, aiMatrix4x4(), name, directory, extension, defaultCullMode, cspace, meshes);

        //for (auto& mesh : meshes) {
        //    ProcessMesh(mesh, scene, directory, extension, defaultCullMode, cspace);
        //}

        std::vector<Async<void>> waiting;
        // There are cases where the number of meshes to process is less than the total available threads,
        // so in that case just use meshes.size() as the upper limit
        const usize numThreads = std::min(INSTANCE(TaskSystem)->Size(), meshes.size());
        std::atomic<usize> counter = 0;
        // Important we start at 1 since we are including this current task thread already
        for (usize i = 1; i < numThreads; ++i) {
            const usize threadNum = i;
            const auto process = [&counter, &meshes, threadNum, numThreads, scene, &directory, &extension, &defaultCullMode, &cspace]() {
                usize processed = 0;
                for (usize idx = threadNum; idx < meshes.size(); idx += numThreads, ++processed) {
                    ProcessMesh(meshes[idx], scene, directory, extension, defaultCullMode, cspace);
                }

                counter += processed;
            };

            waiting.push_back(INSTANCE(TaskSystem)->ScheduleTask(process));
        }

        // Now process our portion
        usize processed = 0;
        for (usize i = 0; i < meshes.size(); i += numThreads, ++processed) {
            ProcessMesh(meshes[i], scene, directory, extension, defaultCullMode, cspace);
        }

        counter += processed;

        while (counter.load() < meshes.size())
            ;

        auto ul = LockWrite_();
        // Create an internal copy for thread safety
        loadedModels_.insert(std::make_pair(name, Async<Entity>(e->Copy())));

        STRATUS_LOG << "Model loaded [" << name << "] with [" << meshes.size() << "] meshes" << std::endl;

        return e->Copy();
    }

    std::shared_ptr<ResourceManager::RawTextureData> ResourceManager::LoadTexture_(const std::vector<std::string>& files, 
                                                                                   const std::vector<BinaryDataWrapper>& binaryData,
                                                                                   const TextureHandle handle, 
                                                                                   const ColorSpace& cspace,
                                                                                   const TextureType type,
                                                                                   const TextureCoordinateWrapping wrap,
                                                                                   const TextureMinificationFilter min,
                                                                                   const TextureMagnificationFilter mag) {

        if (binaryData.size() > 0 && files.size() != binaryData.size()) {
            STRATUS_ERROR << "Invalid file/data length combination" << std::endl;
            return nullptr;
        }

        std::shared_ptr<RawTextureData> texdata = std::make_shared<RawTextureData>();
        texdata->wrap = wrap;
        texdata->min = min;
        texdata->mag = mag;

        #define FREE_ALL_STBI_IMAGE_DATA for (uint8_t * ptr : texdata->data) stbi_image_free((void *)ptr);

        for (usize index = 0; index < files.size(); ++index) {
            const auto& fileOrig = files[index];
            std::string file = fileOrig;
            std::replace(file.begin(), file.end(), '\\', '/');
            STRATUS_LOG << "Attempting to load texture from file: " << file << " (handle = " << handle << ")" << std::endl;

            i32 width, height, numChannels;
            // @see http://www.redbancosdealimentos.org/homes-flooring-design-sources
            u8 * data = nullptr;

            if (binaryData.size() > 0) {
                data = stbi_load_from_memory(binaryData[index].data, (i32)binaryData[index].sizeBytes, &width, &height, &numChannels, 0);
                delete[] binaryData[index].data;
            }
            else {
                data = stbi_load(file.c_str(), &width, &height, &numChannels, 0);
            }

            if (data) {
                // Make sure the width/height match what is already there
                if (texdata->data.size() > 0 &&
                    (texdata->config.width != width || texdata->config.height != height)) {
                        STRATUS_ERROR << "Texture file width/height does not match rest for " << file << std::endl;
                        FREE_ALL_STBI_IMAGE_DATA
                        return nullptr;
                }

                TextureConfig config;
                config.type = type;
                config.storage = TextureComponentSize::BITS_DEFAULT;
                config.generateMipMaps = true;
                config.dataType = TextureComponentType::UINT_NORM;
                config.width = (u32)width;
                config.height = (u32)height;
                config.depth = 0;
                // This loads the textures with sRGB in mind so that they get converted back
                // to linear color space. Warning: if the texture was not actually specified as an
                // sRGB texture (common for normal/specular maps), this will cause problems.
                switch (numChannels) {
                    case 1:
                        config.format = TextureComponentFormat::RED;
                        break;
                    case 3:
                        if (cspace == ColorSpace::SRGB) {
                            config.format = TextureComponentFormat::SRGB;
                        }
                        else {
                            config.format = TextureComponentFormat::RGB;
                        }
                        break;
                    case 4:
                        if (cspace == ColorSpace::SRGB) {
                            config.format = TextureComponentFormat::SRGB_ALPHA;
                        }
                        else {
                            config.format = TextureComponentFormat::RGBA;
                        }
                        break;
                    default:
                        STRATUS_ERROR << "Unknown texture loading error - format may be invalid" << std::endl;
                        FREE_ALL_STBI_IMAGE_DATA
                        return nullptr;
                }

                // Make sure the format type matches what is already there
                if (texdata->data.size() > 0 && texdata->config.format != config.format) {
                    STRATUS_ERROR << "Texture file format does not match rest for " << file << std::endl;
                    FREE_ALL_STBI_IMAGE_DATA
                    return nullptr;
                }

                texdata->config = config;
                texdata->handle = handle;
                texdata->sizeBytes = width * height * numChannels * sizeof(uint8_t);
                texdata->data.push_back(data);
            } 
            else {
                STRATUS_ERROR << "Could not load texture: " << file << std::endl;
                FREE_ALL_STBI_IMAGE_DATA
                return nullptr;
            }
        }

        #undef FREE_ALL_STBI_IMAGE_DATA
    
        // auto ul = _LockWrite();
        // _loadedTextures.insert(std::make_pair(handle, Async<Texture>(*Engine::Instance()->GetMainThread(), [this, texdata]() {
        //     return _FinalizeTexture(*texdata);
        // })));
        return texdata;
    }

    Texture * ResourceManager::FinalizeTexture_(const RawTextureData& data) {
        stratus::TextureArrayData texArrayData(data.data.size());
        for (usize i = 0; i < texArrayData.size(); ++i) {
            texArrayData[i].data = (const void *)data.data[i];
        }
        Texture* texture = new Texture(data.config, texArrayData, false);
        texture->SetHandle_(data.handle);
        texture->SetCoordinateWrapping(data.wrap);
        texture->SetMinMagFilter(data.min, data.mag);
        
        for (uint8_t * ptr : data.data) {
            stbi_image_free((void *)ptr);
        }
        
        return texture;
    }

    EntityPtr ResourceManager::CreateCube() {
        return cube_->Copy();
    }

    EntityPtr ResourceManager::CreateQuad() {
        return quad_->Copy();
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

    void ResourceManager::InitCube_() {
        cube_ = CreateRenderEntity();
        RenderComponent * rc = cube_->Components().GetComponent<RenderComponent>().component;
        MeshPtr mesh = Mesh::Create();
        auto meshlet = mesh->NewMeshlet();
        rc->meshes->meshes.push_back(mesh);
        rc->meshes->transforms.push_back(glm::mat4(1.0f));
        MaterialPtr mat = MaterialManager::Instance()->CreateDefault();
        rc->AddMaterial(mat);

        const usize cubeStride = 14;
        const usize cubeNumVertices = cubeData.size() / cubeStride;

        for (usize i = 0, f = 0; i < cubeNumVertices; ++i, f += cubeStride) {
            meshlet->AddVertex(glm::vec3(cubeData[f], cubeData[f + 1], cubeData[f + 2]));
            meshlet->AddNormal(glm::vec3(cubeData[f + 3], cubeData[f + 4], cubeData[f + 5]));
            meshlet->AddUV(glm::vec2(cubeData[f + 6], cubeData[f + 7]));
            // rmesh->AddTangent(glm::vec3(cubeData[f + 8], cubeData[f + 9], cubeData[f + 10]));
            // rmesh->AddBitangent(glm::vec3(cubeData[f + 11], cubeData[f + 12], cubeData[f + 13]));
        }

        meshlet->CalculateAabbs(glm::mat4(1.0f));
        pendingFinalize_.insert(std::make_pair("DefaultCube", Async<Entity>(cube_)));

        // rmesh->GenerateCpuData();
        // rnode->AddMeshContainer(RenderMeshContainer{rmesh, mat});
        // rmesh->SetFaceCulling(RenderFaceCulling::CULLING_CCW);
        // _pendingFinalize.insert(std::make_pair("DefaultCube", Async<Entity>(_cube)));
        // _cube->SetRenderNode(rnode);
    }

    void ResourceManager::InitQuad_() {
        quad_ = CreateRenderEntity();
        RenderComponent * rc = quad_->Components().GetComponent<RenderComponent>().component;
        MeshPtr mesh = Mesh::Create();
        auto meshlet = mesh->NewMeshlet();
        rc->meshes->meshes.push_back(mesh);
        rc->meshes->transforms.push_back(glm::mat4(1.0f));
        MaterialPtr mat = MaterialManager::Instance()->CreateDefault();
        rc->AddMaterial(mat);

        const usize quadStride = 14;
        const usize quadNumVertices = quadData.size() / quadStride;

        for (usize i = 0, f = 0; i < quadNumVertices; ++i, f += quadStride) {
            meshlet->AddVertex(glm::vec3(quadData[f], quadData[f + 1], quadData[f + 2]));
            meshlet->AddNormal(glm::vec3(quadData[f + 3], quadData[f + 4], quadData[f + 5]));
            meshlet->AddUV(glm::vec2(quadData[f + 6], quadData[f + 7]));
            // rmesh->AddTangent(glm::vec3(quadData[f + 8], quadData[f + 9], quadData[f + 10]));
            // rmesh->AddBitangent(glm::vec3(quadData[f + 11], quadData[f + 12], quadData[f + 13]));
        }

        mesh->SetFaceCulling(RenderFaceCulling::CULLING_NONE);
        meshlet->CalculateAabbs(glm::mat4(1.0f));
        pendingFinalize_.insert(std::make_pair("DefaultQuad", Async<Entity>(quad_)));

        // rmesh->GenerateCpuData();
        // rnode->AddMeshContainer(RenderMeshContainer{rmesh, mat});
        // rmesh->SetFaceCulling(RenderFaceCulling::CULLING_NONE);
        // _pendingFinalize.insert(std::make_pair("DefaultQuad", Async<Entity>(_quad)));
        // _quad->SetRenderNode(rnode);
    }
}