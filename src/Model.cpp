#include "Model.h"
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <iostream>
#include "Renderer.h"

namespace stratus {
    TextureHandle loadMaterialTexture(Renderer & renderer, aiMaterial * mat, const aiTextureType & type, const std::string & directory) {
        TextureHandle texture = -1;

        std::cout << "Tex count: " << mat->GetTextureCount(type) << std::endl;
        if (mat->GetTextureCount(type) > 0) {
            aiString str;
            mat->GetTexture(type, 0, &str);
            std::string file = str.C_Str();
            texture = renderer.loadTexture(directory + "/" + file);
        }

        return texture;
    }

    // @see https://learnopengl.com/Model-Loading/Model
    std::shared_ptr<Mesh> processMesh(Renderer & renderer, aiMesh * mesh, const aiScene * scene, const std::string & directory) {
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
            RenderMaterial mat;
            aiMaterial * material = scene->mMaterials[mesh->mMaterialIndex];
            mat.texture = loadMaterialTexture(renderer, material, aiTextureType_DIFFUSE, directory);
            mat.normalMap = loadMaterialTexture(renderer, material, aiTextureType_NORMALS, directory);
            mat.depthMap = loadMaterialTexture(renderer, material, aiTextureType_HEIGHT, directory);
            mat.roughnessMap = loadMaterialTexture(renderer, material, aiTextureType_SPECULAR, directory);
            mat.environmentMap = loadMaterialTexture(renderer, material, aiTextureType_AMBIENT, directory);
            std::cout << "m " << mat.texture << " "
                << mat.normalMap << " "
                << mat.depthMap << " "
                << mat.roughnessMap << " "
                << mat.environmentMap << std::endl;
            m->setMaterial(mat);
        }

        m->cullingMode = CULLING_CCW;
        return m;
    }  

    static void processNode(Renderer & renderer, aiNode * node, const aiScene * scene, RenderEntity * entity, const std::string & directory) {
        // process all the node's meshes (if any)
        for (uint32_t i = 0; i < node->mNumMeshes; i++) {
            aiMesh * mesh = scene->mMeshes[node->mMeshes[i]];
            entity->meshes.push_back(processMesh(renderer, mesh, scene, directory));
        }

        // then do the same for each of its children
        for (uint32_t i = 0; i < node->mNumChildren; i++) {
            entity->nodes.push_back(RenderEntity(LightProperties::DYNAMIC));
            processNode(renderer, node->mChildren[i], scene, &entity->nodes[entity->nodes.size() - 1], directory);
        }
    }

    Model::Model(Renderer & renderer, const std::string filename) : RenderEntity(LightProperties::DYNAMIC) {
        this->_filename = filename;
        this->_valid = true;
        Assimp::Importer importer;
        //const aiScene *scene = importer.ReadFile(filename, aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_CalcTangentSpace | aiProcess_GenNormals | aiProcess_GenUVCoords);
        const aiScene *scene = importer.ReadFile(filename, aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_FlipUVs | aiProcess_CalcTangentSpace | aiProcess_OptimizeMeshes);

        if(!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
            std::cout << "Error loading model: " << filename << std::endl << importer.GetErrorString() << std::endl;
            this->_valid = false;
            return;
        }

        const std::string directory = filename.substr(0, filename.find_last_of('/'));
        processNode(renderer, scene->mRootNode, scene, this, directory);
    }

    Model::~Model() {}
}