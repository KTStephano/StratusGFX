
#include <RenderEntity.h>

#include "RenderEntity.h"
#include "Common.h"
#include "Utils.h"
#include <iostream>
#include <vector>

namespace stratus {
Mesh::Mesh(const std::vector<glm::vec3> & vertices, const std::vector<glm::vec2> & uvs, const std::vector<glm::vec3> & normals)
    : Mesh(vertices, uvs, normals, {}) {}

Mesh::Mesh(const std::vector<glm::vec3> & vertices, const std::vector<glm::vec2> & uvs, const std::vector<glm::vec3> & normals, const std::vector<uint32_t> & indices) 
    : Mesh(vertices, uvs, normals, {}, {}, indices) {}

Mesh::Mesh(const std::vector<glm::vec3> & vertices, const std::vector<glm::vec2> & uvs, const std::vector<glm::vec3> & normals, const std::vector<glm::vec3> & tangents, const std::vector<glm::vec3> & bitangents, const std::vector<uint32_t> & indices) {
    //assert(vertices.size() % 3 == 0);
    this->_data.data = (void *)&this->_drawData;

    std::vector<uint32_t> indexBuffer;
    const std::vector<uint32_t> * order;
    if (indices.size() == 0) {
        indexBuffer.resize(vertices.size());
        for (uint32_t i = 0; i < vertices.size(); ++i) indexBuffer[i] = i;
        order = &indexBuffer;
    }
    else {
        order = &indices;
    }
    
    // Calculate tangents and bitangents
    // @see https://marti.works/posts/post-calculating-tangents-for-your-mesh/post/
    const std::vector<glm::vec3> * finalizedTangents;
    const std::vector<glm::vec3> * finalizedBitangents;
    std::vector<glm::vec3> computedTangents;
    std::vector<glm::vec3> computedBitangents;

    if (tangents.size() == 0 || bitangents.size() == 0) {
        computedTangents = std::vector<glm::vec3>(vertices.size(), glm::vec3(0.0f));
        computedBitangents = std::vector<glm::vec3>(vertices.size(), glm::vec3(0.0f));
        for (int i = 0; i < order->size(); i += 3) {
            const uint32_t i0 = (*order)[i];
            const uint32_t i1 = (*order)[i + 1];
            const uint32_t i2 = (*order)[i + 2];
            auto tanBitan = calculateTangentAndBitangent(vertices[i0], vertices[i1], vertices[i2], uvs[i0], uvs[i1], uvs[i2]);
            
            computedTangents[i0] += tanBitan.tangent;
            computedTangents[i1] += tanBitan.tangent;
            computedTangents[i2] += tanBitan.tangent;

            computedBitangents[i0] += tanBitan.bitangent;
            computedBitangents[i1] += tanBitan.bitangent;
            computedBitangents[i2] += tanBitan.bitangent;
        }

        for (size_t i = 0; i < vertices.size(); ++i) {
            glm::vec3 & tangent = computedTangents[i];
            glm::vec3 & bitangent = computedBitangents[i];
            const glm::vec3 & normal = normals[i];
            glm::vec3 t = tangent - (normal * glm::dot(normal, tangent));
            t = glm::normalize(t);

            glm::vec3 c = glm::cross(normal, t); // Compute orthogonal 3rd basis vector
            float w = (glm::dot(c, bitangent) < 0) ? -1.0f : 1.0f;
            tangent = t * w;
            bitangent = glm::normalize(c);
        }

        finalizedTangents = &computedTangents;
        finalizedBitangents = &computedBitangents;
    }
    else {
        finalizedTangents = &tangents;
        finalizedBitangents = &bitangents;
    }

    // Pack all data into a single buffer
    std::vector<float> data;
    data.reserve(vertices.size() * 3 + uvs.size() * 2 + normals.size() * 3 + tangents.size() * 3 + bitangents.size() * 3);
    //std::cout << "mesh" << std::endl;
    for (int i = 0; i < vertices.size(); ++i) {
        data.push_back(vertices[i].x);
        data.push_back(vertices[i].y);
        data.push_back(vertices[i].z);
        data.push_back(uvs[i].x);
        data.push_back(uvs[i].y);
        data.push_back(normals[i].x);
        data.push_back(normals[i].y);
        data.push_back(normals[i].z);
        data.push_back((*finalizedTangents)[i].x);
        data.push_back((*finalizedTangents)[i].y);
        data.push_back((*finalizedTangents)[i].z);
        data.push_back((*finalizedBitangents)[i].x);
        data.push_back((*finalizedBitangents)[i].y);
        data.push_back((*finalizedBitangents)[i].z);

/*
        std::cout << vertices[i].x << ", ";
        std::cout << vertices[i].y << ", ";
        std::cout << vertices[i].z << ", ";
        std::cout << uvs[i].x << ", ";
        std::cout << uvs[i].y << ", ";
        std::cout << normals[i].x << ", ";
        std::cout << normals[i].y << ", ";
        std::cout << normals[i].z << ", ";
        std::cout << (*finalizedTangents)[i].x << ", ";
        std::cout << (*finalizedTangents)[i].y << ", ";
        std::cout << (*finalizedTangents)[i].z << ", ";
        std::cout << (*finalizedBitangents)[i].x << ", ";
        std::cout << (*finalizedBitangents)[i].y << ", ";
        std::cout << (*finalizedBitangents)[i].z << std::endl;
*/
    }

    glGenVertexArrays(1, &this->_drawData.vao);
    glGenBuffers(1, &this->_drawData.vbo);
    glGenBuffers(1, &this->_drawData.ebo);

    glBindVertexArray(this->_drawData.vao);
    if (indices.size() > 0) {
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->_drawData.ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(uint32_t), &indices[0], GL_STATIC_DRAW);
    }
    // To get to the next full element we have to skip past a set of vertices (3), uvs (2), normals (3), tangents (3), and bitangents (3)
    const float stride = (3 + 2 + 3 + 3 + 3) * sizeof(float);
    glBindBuffer(GL_ARRAY_BUFFER, this->_drawData.vbo);
    glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(float), &data[0], GL_STATIC_DRAW);
    
    // Vertices
    glEnableVertexAttribArray(0);
    // Index, size, type, normalized?, stride, offset
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void *)0);

    // UVs
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, (void *)(3 * sizeof(float)));
    
    // Normals
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, (void *)(5 * sizeof(float)));

    // Tangents
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, stride, (void *)(8 * sizeof(float)));

    // Bitangents
    glEnableVertexAttribArray(4);
    glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, stride, (void *)(11 * sizeof(float)));

    //glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    //glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    this->_drawData.numVertices = vertices.size();
    this->_drawData.numIndices = indices.size();
}

Mesh::~Mesh() {
    glDeleteVertexArrays(1, &this->_drawData.vao);
    glDeleteBuffers(1, &this->_drawData.vbo);
    glDeleteBuffers(1, &this->_drawData.ebo);
}

void Mesh::_setProperties(uint32_t properties) {
    _properties = (RenderProperties)properties;
}

void Mesh::_enableProperties(uint32_t properties) {
    auto p = (uint32_t)_properties;
    p = p | properties;
    _properties = (RenderProperties)p;
}

void Mesh::_disableProperties(uint32_t properties) {
    auto p = (uint32_t)_properties;
    p = p & (~properties);
    _properties = (RenderProperties)p;
}

void Mesh::setMaterial(const RenderMaterial & material) {
    _material = material;
    _setProperties(RenderProperties::NONE);
    if (material.texture != -1) {
        _enableProperties(TEXTURED);
    }
    if (material.normalMap == -1) {
        _disableProperties(ENVIRONMENT_MAPPED);
        _disableProperties(ROUGHNESS_MAPPED);
        _disableProperties(NORMAL_MAPPED);
        _disableProperties(NORMAL_HEIGHT_MAPPED);
    }
    else {
        if (material.depthMap == -1) {
            _disableProperties(ENVIRONMENT_MAPPED);
            _disableProperties(ROUGHNESS_MAPPED);
            _disableProperties(NORMAL_HEIGHT_MAPPED);
            _enableProperties(NORMAL_MAPPED);
        }
        else {
            _disableProperties(NORMAL_MAPPED);
            _enableProperties(NORMAL_HEIGHT_MAPPED);

            // Only want to set this if both normal and depth mapping are available
            if (material.roughnessMap != -1) {
                _enableProperties(ROUGHNESS_MAPPED);

                if (material.environmentMap != -1) {
                    _enableProperties(ENVIRONMENT_MAPPED);
                }
            }
        }
    }
}

const RenderProperties & Mesh::getRenderProperties() const {
    return _properties;
}

const RenderMaterial & Mesh::getMaterial() const {
    return _material;
}

const RenderData & Mesh::getRenderData() const {
    return _data;
}

void Mesh::render(const int numInstances) const {    
    if (this->_drawData.numIndices > 0) {
        glDrawElementsInstanced(GL_TRIANGLES, this->_drawData.numIndices, GL_UNSIGNED_INT, (void *)0, numInstances);
    }
    else {
        //std::cout << "glDrawArraysInstanced(GL_TRIANGLES, 0, " << this->_drawData.numVertices << ", " << numInstances << ");" << std::endl;
        glDrawArraysInstanced(GL_TRIANGLES, 0, this->_drawData.numVertices, numInstances);
    }
}

void Mesh::bind() const {
    glBindVertexArray(this->_drawData.vao);
}

void Mesh::unbind() const {
    glBindVertexArray(0);
}

size_t Mesh::hashCode() const {
    return std::hash<void *>{}(getRenderData().data) +
        std::hash<int>{}(int(getRenderProperties())) +
        std::hash<int>{}(getMaterial().texture) +
        std::hash<int>{}(getMaterial().normalMap) +
        std::hash<int>{}(getMaterial().depthMap) +
        std::hash<int>{}(getMaterial().roughnessMap) +
        std::hash<int>{}(getMaterial().environmentMap) +
        std::hash<int>{}(cullingMode);
}

bool Mesh::operator==(const Mesh & m) const {
    return getRenderData().data == m.getRenderData().data &&
        getRenderProperties() == m.getRenderProperties() &&
        getMaterial().texture == m.getMaterial().texture &&
        getMaterial().normalMap == m.getMaterial().normalMap &&
        getMaterial().depthMap == m.getMaterial().depthMap &&
        getMaterial().roughnessMap == m.getMaterial().roughnessMap &&
        getMaterial().environmentMap == m.getMaterial().environmentMap &&
        cullingMode == m.cullingMode;
}

RenderEntity::RenderEntity(LightProperties properties) {
    setLightProperties(properties);
}

RenderEntity::~RenderEntity() {

}

void RenderEntity::setLightProperties(const LightProperties & properties) {
    this->_properties = properties;
    for (auto & node : nodes) {
        node.setLightProperties(properties);
    }
}

const LightProperties & RenderEntity::getLightProperties() const {
    return this->_properties;
}

size_t RenderEntity::hashCode() const {
    return std::hash<int>{}(getLightProperties());
}

bool RenderEntity::operator==(const RenderEntity & e) const {
    return getLightProperties() == e.getLightProperties();
}
}