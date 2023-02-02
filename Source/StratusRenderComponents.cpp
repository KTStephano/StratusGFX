#include "StratusRenderComponents.h"

#include "StratusUtils.h"
#include "StratusEntity.h"
#include "StratusApplicationThread.h"
#include "StratusLog.h"
#include "StratusTransformComponent.h"
#include "StratusPoolAllocator.h"

namespace stratus {
    struct MeshAllocator {
        std::mutex m;
        PoolAllocator<Mesh> allocator;
    };

    static MeshAllocator& GetAllocator() {
        static MeshAllocator allocator;
        return allocator;
    }

    MeshPtr Mesh::_PlacementNew(uint8_t * memory) {
        return new (memory) Mesh();
    }

    MeshPtr Mesh::Create() {
        auto& allocator = GetAllocator();
        auto ul = std::unique_lock<std::mutex>(allocator.m);
        return allocator.allocator.AllocateCustomConstruct(_PlacementNew);
    }

    void Mesh::Destroy(MeshPtr ptr) {
        auto& allocator = GetAllocator();
        auto ul = std::unique_lock<std::mutex>(allocator.m);
        allocator.allocator.Deallocate(ptr);
    }

    Entity2Ptr CreateRenderEntity() {
        auto ptr = Entity2::Create();
        InitializeRenderEntity(ptr);
        return ptr;
    }

    void InitializeRenderEntity(const Entity2Ptr& ptr) {
        InitializeTransformEntity(ptr);
        ptr->Components().AttachComponent<RenderComponent>();
        ptr->Components().AttachComponent<LightInteractionComponent>();
        ptr->Components().AttachComponent<StaticObjectComponent>();
    }

    Mesh::Mesh() {
        _cpuData = new _MeshCpuData();
    }

    Mesh::~Mesh() {
        delete _cpuData;
        _cpuData = nullptr;
    }

    bool Mesh::IsFinalized() const {
        return _cpuData == nullptr;
    }

    void Mesh::_EnsureFinalized() const {
        if (!IsFinalized()) {
            throw std::runtime_error("Attempt to read GPU Mesh data before finalized");
        }
    }

    void Mesh::_EnsureNotFinalized() const {
        if (IsFinalized()) {
            throw std::runtime_error("Attempt to write GPU Mesh data after finalized");
        }
    }

    void Mesh::AddVertex(const glm::vec3& v) {
        _EnsureNotFinalized();
        _cpuData->vertices.push_back(v);
        _numVertices = _cpuData->vertices.size();
    }

    void Mesh::AddUV(const glm::vec2& uv) {
        _EnsureNotFinalized();
        _cpuData->uvs.push_back(uv);
    }

    void Mesh::AddNormal(const glm::vec3& n) {
        _EnsureNotFinalized();
        _cpuData->normals.push_back(n);
    }

    void Mesh::AddTangent(const glm::vec3& t) {
        _EnsureNotFinalized();
        _cpuData->tangents.push_back(t);
    }

    void Mesh::AddBitangent(const glm::vec3& bt) {
        _EnsureNotFinalized();
        _cpuData->bitangents.push_back(bt);
    }

    void Mesh::AddIndex(uint32_t i) {
        _EnsureNotFinalized();
        _cpuData->indices.push_back(i);
        _numIndices = _cpuData->indices.size();
    }

    void Mesh::_CalculateTangentsBitangents() {
        _EnsureNotFinalized();
        _cpuData->tangents.clear();
        _cpuData->bitangents.clear();

        std::vector<uint32_t> indexBuffer;
        const std::vector<uint32_t> * order;
        if (_numIndices == 0) {
            indexBuffer.resize(_numVertices);
            for (uint32_t i = 0; i < _numVertices; ++i) indexBuffer[i] = i;
            order = &indexBuffer;
        }
        else {
            order = &_cpuData->indices;
        }

        _cpuData->tangents = std::vector<glm::vec3>(_numVertices, glm::vec3(0.0f));
        _cpuData->bitangents = std::vector<glm::vec3>(_numVertices, glm::vec3(0.0f));
        for (int i = 0; i < order->size(); i += 3) {
            const uint32_t i0 = (*order)[i];
            const uint32_t i1 = (*order)[i + 1];
            const uint32_t i2 = (*order)[i + 2];
            auto tanBitan = calculateTangentAndBitangent(_cpuData->vertices[i0], _cpuData->vertices[i1], _cpuData->vertices[i2], _cpuData->uvs[i0], _cpuData->uvs[i1], _cpuData->uvs[i2]);
            
            _cpuData->tangents[i0] += tanBitan.tangent;
            _cpuData->tangents[i1] += tanBitan.tangent;
            _cpuData->tangents[i2] += tanBitan.tangent;

            _cpuData->bitangents[i0] += tanBitan.bitangent;
            _cpuData->bitangents[i1] += tanBitan.bitangent;
            _cpuData->bitangents[i2] += tanBitan.bitangent;
        }

        for (size_t i = 0; i < _numVertices; ++i) {
            glm::vec3 & tangent = _cpuData->tangents[i];
            glm::vec3 & bitangent = _cpuData->bitangents[i];
            const glm::vec3 & normal = _cpuData->normals[i];
            glm::vec3 t = tangent - (normal * glm::dot(normal, tangent));
            t = glm::normalize(t);

            glm::vec3 c = glm::cross(normal, t); // Compute orthogonal 3rd basis vector
            float w = (glm::dot(c, bitangent) < 0) ? -1.0f : 1.0f;
            tangent = t * w;
            bitangent = glm::normalize(c);
        }
    }

    void Mesh::_GenerateCpuData() {
        _EnsureNotFinalized();

        if (_cpuData->tangents.size() == 0 || _cpuData->bitangents.size() == 0) _CalculateTangentsBitangents();

        // Pack all data into a single buffer
        _cpuData->data.clear();
        _cpuData->data.reserve(_cpuData->vertices.size() * 3 + _cpuData->uvs.size() * 2 + _cpuData->normals.size() * 3 + _cpuData->tangents.size() * 3 + _cpuData->bitangents.size() * 3);
        for (int i = 0; i < _numVertices; ++i) {
            _cpuData->data.push_back(_cpuData->vertices[i].x);
            _cpuData->data.push_back(_cpuData->vertices[i].y);
            _cpuData->data.push_back(_cpuData->vertices[i].z);
            _cpuData->data.push_back(_cpuData->uvs[i].x);
            _cpuData->data.push_back(_cpuData->uvs[i].y);
            _cpuData->data.push_back(_cpuData->normals[i].x);
            _cpuData->data.push_back(_cpuData->normals[i].y);
            _cpuData->data.push_back(_cpuData->normals[i].z);
            _cpuData->data.push_back(_cpuData->tangents[i].x);
            _cpuData->data.push_back(_cpuData->tangents[i].y);
            _cpuData->data.push_back(_cpuData->tangents[i].z);
            _cpuData->data.push_back(_cpuData->bitangents[i].x);
            _cpuData->data.push_back(_cpuData->bitangents[i].y);
            _cpuData->data.push_back(_cpuData->bitangents[i].z);
        }

        _dataSizeBytes = _cpuData->data.size() * sizeof(float);

        // Clear out existing buffers to conserve system memory
        _cpuData->vertices.clear();
        _cpuData->uvs.clear();
        _cpuData->normals.clear();
        _cpuData->tangents.clear();
        _cpuData->bitangents.clear();
    }

    size_t Mesh::GetGpuSizeBytes() const {
        _EnsureFinalized();
        return _dataSizeBytes;
    }

    void Mesh::_GenerateGpuData() {
        _EnsureNotFinalized();

        _buffers = GpuArrayBuffer();
        GpuPrimitiveBuffer buffer;
        if (_numIndices > 0) {
            buffer = GpuPrimitiveBuffer(GpuPrimitiveBindingPoint::ELEMENT_ARRAY_BUFFER, _cpuData->indices.data(), _cpuData->indices.size() * sizeof(uint32_t));
            _buffers.AddBuffer(buffer);
            //_cpuData->indicesMapped = buffer.MapMemory();
        }

        // To get to the next full element we have to skip past a set of vertices (3), uvs (2), normals (3), tangents (3), and bitangents (3)
        buffer = GpuPrimitiveBuffer(GpuPrimitiveBindingPoint::ARRAY_BUFFER, _cpuData->data.data(), _cpuData->data.size() * sizeof(float));
        _buffers.AddBuffer(buffer);
        //_primitiveMapped = buffer.MapMemory();
        
        const float stride = (3 + 2 + 3 + 3 + 3) * sizeof(float);

        // Vertices
        buffer.EnableAttribute(0, 3, GpuStorageType::FLOAT, false, stride, 0);

        // UVs
        buffer.EnableAttribute(1, 2, GpuStorageType::FLOAT, false, stride, 3 * sizeof(float));

        // Normals
        buffer.EnableAttribute(2, 3, GpuStorageType::FLOAT, false, stride, 5 * sizeof(float));

        // Tangents
        buffer.EnableAttribute(3, 3, GpuStorageType::FLOAT, false, stride, 8 * sizeof(float));

        // Bitangents
        buffer.EnableAttribute(4, 3, GpuStorageType::FLOAT, false, stride, 11 * sizeof(float));

        // Clear CPU memory
        delete _cpuData;
        _cpuData = nullptr;
    }

    void Mesh::FinalizeData() {
        _EnsureNotFinalized();

        _GenerateCpuData();

        if (ApplicationThread::Instance()->CurrentIsApplicationThread()) {
            _GenerateGpuData();
            _buffers.UnmapAllMemory();
            _buffers.FinalizeAllMemory();
        }
        else {
            ApplicationThread::Instance()->Queue([this]() {
                _GenerateGpuData();
                _buffers.UnmapAllMemory();
                _buffers.FinalizeAllMemory();
            });
        }
    }

    const GpuArrayBuffer& Mesh::GetData() const {
        _EnsureFinalized();
        return _buffers;
    }

    // void Mesh::Render(size_t numInstances, const GpuArrayBuffer& additionalBuffers) const {
    //     // GenerateCpuData();
    //     // GenerateGpuData();
    //     // FinalizeGpuData();

    //     // We don't want to run if our memory is mapped since another thread might be
    //     // writing data to a buffer's memory region
    //     if (!Complete()) return;

    //     //if (_primitiveMapped != nullptr || _cpuData->indicesMapped != nullptr) {
    //     //    //_buffers.UnmapAllReadWrite();
    //     //    _primitiveMapped = nullptr;
    //     //    _cpuData->indicesMapped = nullptr;
    //     //}

    //     _buffers.Bind();
    //     additionalBuffers.Bind();

    //     if (_numIndices > 0) {
    //         glDrawElementsInstanced(GL_TRIANGLES, _numIndices, GL_UNSIGNED_INT, (void *)0, numInstances);
    //     }
    //     else {
    //         glDrawArraysInstanced(GL_TRIANGLES, 0, _numVertices, numInstances);
    //     }

    //     additionalBuffers.Unbind();
    //     _buffers.Unbind();
    // }

    void Mesh::SetFaceCulling(const RenderFaceCulling& cullMode) {
        _cullMode = cullMode;
    }

    RenderFaceCulling Mesh::GetFaceCulling() const {
        return _cullMode;
    }

    RenderComponent::RenderComponent()
        : meshes(std::make_shared<MeshData>()) {}

    RenderComponent::RenderComponent(const RenderComponent& other) {
        this->meshes = other.meshes;
    }

    size_t RenderComponent::NumMaterials() const {
        return _materials.size();
    }

    const std::vector<MaterialPtr>& RenderComponent::GetAllMaterials() const {
        return _materials;
    }

    const MaterialPtr& RenderComponent::GetMaterialAt(size_t index) const {
        return _materials[index];
    }

    void RenderComponent::AddMaterial(MaterialPtr material) {
        _materials.push_back(material);
    }

    void RenderComponent::SetMaterialAt(MaterialPtr material, size_t index) {
        _materials[index] = material;
    }
}