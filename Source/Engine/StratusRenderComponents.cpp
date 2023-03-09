#include "StratusRenderComponents.h"

#include "StratusUtils.h"
#include "StratusEntity.h"
#include "StratusApplicationThread.h"
#include "StratusLog.h"
#include "StratusTransformComponent.h"
#include "StratusPoolAllocator.h"
#include "meshoptimizer.h"

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

    EntityPtr CreateRenderEntity() {
        auto ptr = Entity::Create();
        InitializeRenderEntity(ptr);
        return ptr;
    }

    void InitializeRenderEntity(const EntityPtr& ptr) {
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

        auto vertexOffset = _vertexOffset;
        auto numVertices = _numVertices;
        auto indexOffsetPerLod = _indexOffsetPerLod;
        auto numIndicesPerLod = _numIndicesPerLod;
        const auto deallocate = [vertexOffset, numVertices, indexOffsetPerLod, numIndicesPerLod]() {
            GpuMeshAllocator::DeallocateVertexData(vertexOffset, numVertices);
            for (size_t i = 0; i < indexOffsetPerLod.size(); ++i) {
                GpuMeshAllocator::DeallocateIndexData(indexOffsetPerLod[i], numIndicesPerLod[i]);
            }
        };

        if (ApplicationThread::Instance()->CurrentIsApplicationThread()) {
            deallocate();
        }
        else {
            ApplicationThread::Instance()->Queue(deallocate);
        }
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
        _cpuData->needsRepacking = true;
        _numVertices = _cpuData->vertices.size();
    }

    void Mesh::AddUV(const glm::vec2& uv) {
        _EnsureNotFinalized();
        _cpuData->uvs.push_back(uv);
        _cpuData->needsRepacking = true;
    }

    void Mesh::AddNormal(const glm::vec3& n) {
        _EnsureNotFinalized();
        _cpuData->normals.push_back(n);
        _cpuData->needsRepacking = true;
    }

    void Mesh::AddTangent(const glm::vec3& t) {
        _EnsureNotFinalized();
        _cpuData->tangents.push_back(t);
        _cpuData->needsRepacking = true;
    }

    void Mesh::AddBitangent(const glm::vec3& bt) {
        _EnsureNotFinalized();
        _cpuData->bitangents.push_back(bt);
        _cpuData->needsRepacking = true;
    }

    void Mesh::AddIndex(uint32_t i) {
        _EnsureNotFinalized();
        _cpuData->indices.push_back(i);
        _numIndices = _cpuData->indices.size();
    }

    void Mesh::_CalculateTangentsBitangents() {
        _EnsureNotFinalized();
        _cpuData->needsRepacking = true;
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

    void Mesh::PackCpuData() {
        _EnsureNotFinalized();

        if (_cpuData->tangents.size() == 0 || _cpuData->bitangents.size() == 0) _CalculateTangentsBitangents();

        if (!_cpuData->needsRepacking) return;

        // Pack all data into a single buffer
        _cpuData->data.clear();
        _cpuData->data.resize(_numVertices);
        //_cpuData->data.reserve(_cpuData->vertices.size() * 3 + _cpuData->uvs.size() * 2 + _cpuData->normals.size() * 3 + _cpuData->tangents.size() * 3 + _cpuData->bitangents.size() * 3);
        for (int i = 0; i < _numVertices; ++i) {
            // _cpuData->data.push_back(_cpuData->vertices[i].x);
            // _cpuData->data.push_back(_cpuData->vertices[i].y);
            // _cpuData->data.push_back(_cpuData->vertices[i].z);
            // _cpuData->data.push_back(_cpuData->uvs[i].x);
            // _cpuData->data.push_back(_cpuData->uvs[i].y);
            // _cpuData->data.push_back(_cpuData->normals[i].x);
            // _cpuData->data.push_back(_cpuData->normals[i].y);
            // _cpuData->data.push_back(_cpuData->normals[i].z);
            // _cpuData->data.push_back(_cpuData->tangents[i].x);
            // _cpuData->data.push_back(_cpuData->tangents[i].y);
            // _cpuData->data.push_back(_cpuData->tangents[i].z);
            // _cpuData->data.push_back(_cpuData->bitangents[i].x);
            // _cpuData->data.push_back(_cpuData->bitangents[i].y);
            // _cpuData->data.push_back(_cpuData->bitangents[i].z);
            GpuMeshData * data = &_cpuData->data[i];
            data->position[0] = _cpuData->vertices[i].x;
            data->position[1] = _cpuData->vertices[i].y;
            data->position[2] = _cpuData->vertices[i].z;
            data->texCoord[0] = _cpuData->uvs[i].x;
            data->texCoord[1] = _cpuData->uvs[i].y;
            data->normal[0] = _cpuData->normals[i].x;
            data->normal[1] = _cpuData->normals[i].y;
            data->normal[2] = _cpuData->normals[i].z;
            data->tangent[0] = _cpuData->tangents[i].x;
            data->tangent[1] = _cpuData->tangents[i].y;
            data->tangent[2] = _cpuData->tangents[i].z;
            data->bitangent[0] = _cpuData->bitangents[i].x;
            data->bitangent[1] = _cpuData->bitangents[i].y;
            data->bitangent[2] = _cpuData->bitangents[i].z;
        }

        _dataSizeBytes = _cpuData->data.size() * sizeof(GpuMeshData);

        _cpuData->needsRepacking = false;
    }

    // This comes from the Visibility and Occlusion chapter in "Foundations of Game Engine Development, Volume 2: Rendering"
    void Mesh::CalculateAabbs(const glm::mat4& transform) {
        assert(_cpuData->vertices.size() > 0);
        _EnsureNotFinalized();

        // If no indices generate a buffer from [0, num vertices)
        // This does not require CPU data to be repacked
        if (_cpuData->indices.size() == 0) {
            for (uint32_t i = 0; i < _numVertices; ++i) {
                AddIndex(i);
            }
        }

        glm::vec3 vertex = glm::vec3(transform * glm::vec4(_cpuData->vertices[_cpuData->indices[0]], 1.0f));
		glm::vec3 vmin = vertex;
		glm::vec3 vmax = vertex;
        for (size_t i = 0; i < _cpuData->indices.size(); ++i) {
            vertex = glm::vec3(transform * glm::vec4(_cpuData->vertices[_cpuData->indices[i]], 1.0f));
            vmin = glm::min(vmin, vertex);
            vmax = glm::max(vmax, vertex);
        }

        _aabb.vmin = glm::vec4(vmin, 1.0f);
        _aabb.vmax = glm::vec4(vmax, 1.0f);
        //aabb.center = (vmin + vmax) * 0.5f;
        //aabb.size = (vmax - vmin) * 0.5f;
    }

    size_t Mesh::GetGpuSizeBytes() const {
        _EnsureFinalized();
        return _dataSizeBytes;
    }

    const GpuAABB& Mesh::GetAABB() const {
        _EnsureFinalized();
        return _aabb;
    }

    uint32_t Mesh::GetVertexOffset() const {
        return _vertexOffset;
    }

    uint32_t Mesh::GetIndexOffset(size_t lod) const {
        lod = lod >= _indexOffsetPerLod.size() ? _indexOffsetPerLod.size() - 1 : lod;
        return _indexOffsetPerLod[lod];
    }

    uint32_t Mesh::GetNumIndices(size_t lod) const {
        lod = lod >= _numIndicesPerLod.size() ? _numIndicesPerLod.size() - 1 : lod;
        return _numIndicesPerLod[lod];
    }

    void Mesh::_GenerateGpuData() {
        _EnsureNotFinalized();

        // If no indices generate a buffer from [0, num vertices)
        // This does not require CPU data to be repacked
        if (_cpuData->indices.size() == 0) {
            for (uint32_t i = 0; i < _numVertices; ++i) {
                AddIndex(i);
            }
        }

        std::vector<std::vector<uint32_t>> indicesPerLod;
        indicesPerLod.push_back(_cpuData->indices);
        _numIndicesPerLod.push_back(_cpuData->indices.size());
        for (int i = 0; i < 7; ++i) {
            auto& prevIndices = indicesPerLod[indicesPerLod.size() - 1];
            std::vector<uint32_t> simplified(prevIndices.size());
            auto size = meshopt_simplify(simplified.data(), prevIndices.data(), prevIndices.size(), &_cpuData->vertices[0][0], _numVertices, sizeof(float) * 3, prevIndices.size() / 2, 0.01f);
            // If we didn't see at least a 10% reduction, try the more aggressive algorithm
            if ((prevIndices.size() * 0.9) < double(size)) {
                size = meshopt_simplifySloppy(simplified.data(), prevIndices.data(), prevIndices.size(), &_cpuData->vertices[0][0], _numVertices, sizeof(float) * 3, prevIndices.size() / 2, 0.01f);
            }
            simplified.resize(size);
            indicesPerLod.push_back(std::move(simplified));
            _numIndicesPerLod.push_back(size);
            if (size < 1024) break;
        }

        _vertexOffset = GpuMeshAllocator::AllocateVertexData(_numVertices);

        for (auto& indices : indicesPerLod) {
            _indexOffsetPerLod.push_back(GpuMeshAllocator::AllocateIndexData(indices.size()));
            // Account for the fact that all vertices are stored in a global GpuBuffer and so
            // the indices need to be offset
            for (size_t i = 0; i < indices.size(); ++i) {
                indices[i] += _vertexOffset;
            }
            
            GpuMeshAllocator::CopyIndexData(indices, _indexOffsetPerLod[_indexOffsetPerLod.size() - 1]);
        }

        GpuMeshAllocator::CopyVertexData(_cpuData->data, _vertexOffset);

        //_meshData = GpuBuffer((const void *)_cpuData->data.data(), _dataSizeBytes, GPU_MAP_READ);
        //_indices = GpuPrimitiveBuffer(GpuPrimitiveBindingPoint::ELEMENT_ARRAY_BUFFER, _cpuData->indices.data(), _cpuData->indices.size() * sizeof(uint32_t));
        //_buffers.AddBuffer(buffer);
        //_cpuData->indicesMapped = buffer.MapMemory();

        // To get to the next full element we have to skip past a set of vertices (3), uvs (2), normals (3), tangents (3), and bitangents (3)
        // buffer = GpuPrimitiveBuffer(GpuPrimitiveBindingPoint::ARRAY_BUFFER, _cpuData->data.data(), _cpuData->data.size() * sizeof(float));
        // _buffers.AddBuffer(buffer);
        // //_primitiveMapped = buffer.MapMemory();
        
        // const float stride = (3 + 2 + 3 + 3 + 3) * sizeof(float);

        // // Vertices
        // buffer.EnableAttribute(0, 3, GpuStorageType::FLOAT, false, stride, 0);

        // // UVs
        // buffer.EnableAttribute(1, 2, GpuStorageType::FLOAT, false, stride, 3 * sizeof(float));

        // // Normals
        // buffer.EnableAttribute(2, 3, GpuStorageType::FLOAT, false, stride, 5 * sizeof(float));

        // // Tangents
        // buffer.EnableAttribute(3, 3, GpuStorageType::FLOAT, false, stride, 8 * sizeof(float));

        // // Bitangents
        // buffer.EnableAttribute(4, 3, GpuStorageType::FLOAT, false, stride, 11 * sizeof(float));

        // Clear CPU memory
        delete _cpuData;
        _cpuData = nullptr;
    }

    void Mesh::FinalizeData() {
        _EnsureNotFinalized();

        PackCpuData();

        if (ApplicationThread::Instance()->CurrentIsApplicationThread()) {
            _GenerateGpuData();
        }
        else {
            ApplicationThread::Instance()->Queue([this]() {
                _GenerateGpuData();
            });
        }
    }

    void Mesh::Render(size_t numInstances, const GpuArrayBuffer& additionalBuffers) const {
        if (!IsFinalized()) return;

        //if (_primitiveMapped != nullptr || _cpuData->indicesMapped != nullptr) {
        //    //_buffers.UnmapAllReadWrite();
        //    _primitiveMapped = nullptr;
        //    _cpuData->indicesMapped = nullptr;
        //}

        // Matches the location in mesh_data.glsl
        additionalBuffers.Bind();

        glDrawElementsInstanced(GL_TRIANGLES, _numIndices, GL_UNSIGNED_INT, (const void *)(GetIndexOffset(0) * sizeof(uint32_t)), numInstances);

        additionalBuffers.Unbind();
        //GpuMeshAllocator::UnbindElementArrayBuffer();
    }

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
        this->_materials = other._materials;
    }

    MeshPtr RenderComponent::GetMesh(const size_t meshIndex) const {
        return meshes->meshes[meshIndex];
    }

    size_t RenderComponent::GetMeshCount() const {
        return meshes->meshes.size();
    }

    const glm::mat4& RenderComponent::GetMeshTransform(const size_t meshIndex) const {
        return meshes->transforms[meshIndex];
    }

    size_t RenderComponent::GetMaterialCount() const {
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
        MarkChanged();
    }

    void RenderComponent::SetMaterialAt(MaterialPtr material, size_t index) {
        _materials[index] = material;
        MarkChanged();
    }
}