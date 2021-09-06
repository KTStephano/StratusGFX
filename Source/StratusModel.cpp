#include "StratusModel.h"
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <iostream>
#include "StratusRenderer.h"
#include "StratusUtils.h"
#include "StratusLog.h"

namespace stratus {
    TextureHandle loadMaterialTexture(Renderer & renderer, aiMaterial * mat, const aiTextureType & type, const std::string & directory) {
        TextureHandle texture;

        STRATUS_LOG << "Tex count: " << mat->GetTextureCount(type) << std::endl;
        if (mat->GetTextureCount(type) > 0) {
            aiString str;
            mat->GetTexture(type, 0, &str);
            std::string file = str.C_Str();
            STRATUS_LOG << file << std::endl;
            texture = renderer.loadTexture(directory + "/" + file);
        }

        return texture;
    }

    // @see https://learnopengl.com/Model-Loading/Model
    std::shared_ptr<Mesh> processMesh(Renderer & renderer, aiMesh * mesh, const aiScene * scene, MaterialPtr rootMat, const std::string & directory) {
        if (mesh->mNumUVComponents[0] == 0) return nullptr;
        std::vector<glm::vec3> vertices;
        std::vector<glm::vec2> uvs;
        std::vector<glm::vec3> normals;
        std::vector<glm::vec3> tangents;
        std::vector<glm::vec3> bitangents;
        std::vector<uint32_t> indices;

        for (uint32_t i = 0; i < mesh->mNumVertices; i++) {
            vertices.push_back(glm::vec3(mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z));
            uvs.push_back(glm::vec2(mesh->mTextureCoords[0][i].x, mesh->mTextureCoords[0][i].y));
            normals.push_back(glm::vec3(mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z));
            tangents.push_back(glm::vec3(mesh->mTangents[i].x, mesh->mTangents[i].y, mesh->mTangents[i].z));
            bitangents.push_back(glm::vec3(mesh->mBitangents[i].x, mesh->mBitangents[i].y, mesh->mBitangents[i].z));
        }

        // process indices
        for(uint32_t i = 0; i < mesh->mNumFaces; i++) {
            aiFace face = mesh->mFaces[i];
            for(uint32_t j = 0; j < face.mNumIndices; j++) {
                indices.push_back(face.mIndices[j]);
            }
        }  

        std::shared_ptr<Mesh> m = std::make_shared<Mesh>(vertices, uvs, normals, tangents, bitangents, indices);
        // process material
        if (mesh->mMaterialIndex >= 0) {
            // Only support one material for now
            MaterialPtr mat = rootMat->CreateSubMaterial();
            aiMaterial * material = scene->mMaterials[mesh->mMaterialIndex];

            STRATUS_LOG << "mat\n";
            STRATUS_LOG << material->GetTextureCount(aiTextureType_DIFFUSE) << std::endl;
            STRATUS_LOG << material->GetTextureCount(aiTextureType_SPECULAR) << std::endl;
            STRATUS_LOG << material->GetTextureCount(aiTextureType_AMBIENT) << std::endl;
            STRATUS_LOG << material->GetTextureCount(aiTextureType_EMISSIVE) << std::endl;
            STRATUS_LOG << material->GetTextureCount(aiTextureType_HEIGHT) << std::endl;
            STRATUS_LOG << material->GetTextureCount(aiTextureType_NORMALS) << std::endl;
            STRATUS_LOG << material->GetTextureCount(aiTextureType_SHININESS) << std::endl;
            STRATUS_LOG << material->GetTextureCount(aiTextureType_OPACITY) << std::endl;
            STRATUS_LOG << material->GetTextureCount(aiTextureType_DISPLACEMENT) << std::endl;
            STRATUS_LOG << material->GetTextureCount(aiTextureType_LIGHTMAP) << std::endl;
            STRATUS_LOG << material->GetTextureCount(aiTextureType_REFLECTION) << std::endl;

            mat->SetDiffuseTexture(loadMaterialTexture(renderer, material, aiTextureType_DIFFUSE, directory));
            mat->SetNormalMap(loadMaterialTexture(renderer, material, aiTextureType_NORMALS, directory));
            mat->SetDepthMap(loadMaterialTexture(renderer, material, aiTextureType_HEIGHT, directory));
            mat->SetRoughnessMap(loadMaterialTexture(renderer, material, aiTextureType_SHININESS, directory));
            mat->SetAmbientTexture(loadMaterialTexture(renderer, material, aiTextureType_AMBIENT, directory));
            mat->SetMetallicMap(loadMaterialTexture(renderer, material, aiTextureType_SPECULAR, directory));
            STRATUS_LOG << "m " << mat->GetDiffuseTexture() << " "
                << mat->GetNormalMap() << " "
                << mat->GetDepthMap() << " "
                << mat->GetRoughnessMap() << " "
                << mat->GetAmbientTexture() << " "
                << mat->GetMetallicMap() << std::endl;
            m->setMaterial(mat);
        }

        m->cullingMode = CULLING_CCW;
        return m;
    }  

    static void processNode(Renderer & renderer, aiNode * node, const aiScene * scene, RenderEntity * entity, MaterialPtr rootMat, const std::string & directory) {
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

        entity->scale = glm::vec3(scale.x, scale.y, scale.z);
        entity->rotation = Rotation(angleX, angleY, angleZ);
        entity->position = glm::vec3(position.x, position.y, position.z);

        // process all the node's meshes (if any)
        for (uint32_t i = 0; i < node->mNumMeshes; i++) {
            aiMesh * mesh = scene->mMeshes[node->mMeshes[i]];
            auto sm = processMesh(renderer, mesh, scene, rootMat, directory);
            if (!sm) continue;
            entity->meshes.push_back(sm);
        }

        // then do the same for each of its children
        for (uint32_t i = 0; i < node->mNumChildren; i++) {
            entity->nodes.push_back(RenderEntity(LightProperties::DYNAMIC));
            processNode(renderer, node->mChildren[i], scene, &entity->nodes[entity->nodes.size() - 1], rootMat, directory);
        }
    }

    Model::Model(Renderer & renderer, const std::string filename) : RenderEntity(LightProperties::DYNAMIC) {
        this->_filename = filename;
        this->_valid = true;
        Assimp::Importer importer;
        //const aiScene *scene = importer.ReadFile(filename, aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_CalcTangentSpace | aiProcess_GenNormals | aiProcess_GenUVCoords);
        //const aiScene *scene = importer.ReadFile(filename, aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_FlipUVs | aiProcess_CalcTangentSpace | aiProcess_OptimizeMeshes);
        const aiScene *scene = importer.ReadFile(filename, aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_FlipUVs | aiProcess_GenUVCoords | aiProcess_CalcTangentSpace | aiProcess_SplitLargeMeshes);

        auto material = MaterialManager::Instance()->CreateMaterial(filename);

        if(!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
            STRATUS_ERROR << "Error loading model: " << filename << std::endl << importer.GetErrorString() << std::endl;
            this->_valid = false;
            return;
        }

        const std::string directory = filename.substr(0, filename.find_last_of('/'));
        processNode(renderer, scene->mRootNode, scene, this, material, directory);
    }

    Model::~Model() {}
}