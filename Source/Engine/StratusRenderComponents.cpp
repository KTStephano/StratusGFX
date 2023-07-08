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

    MeshPtr Mesh::PlacementNew_(uint8_t * memory) {
        return new (memory) Mesh();
    }

    MeshPtr Mesh::Create() {
        auto& allocator = GetAllocator();
        auto ul = std::unique_lock<std::mutex>(allocator.m);
        return allocator.allocator.AllocateCustomConstruct(PlacementNew_);
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
        cpuData_ = new MeshCpuData_();
    }

    Mesh::~Mesh() {
        delete cpuData_;
        cpuData_ = nullptr;

        auto vertexOffset = vertexOffset_;
        auto numVertices = numVertices_;
        auto indexOffsetPerLod = indexOffsetPerLod_;
        auto numIndicesPerLod = numIndicesPerLod_;
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
        return cpuData_ == nullptr;
    }

    void Mesh::EnsureFinalized_() const {
        if (!IsFinalized()) {
            throw std::runtime_error("Attempt to read GPU Mesh data before finalized");
        }
    }

    void Mesh::EnsureNotFinalized_() const {
        if (IsFinalized()) {
            throw std::runtime_error("Attempt to write GPU Mesh data after finalized");
        }
    }

    void Mesh::AddVertex(const glm::vec3& v) {
        EnsureNotFinalized_();
        cpuData_->vertices.push_back(v);
        cpuData_->needsRepacking = true;
        numVertices_ = cpuData_->vertices.size();
    }

    void Mesh::AddUV(const glm::vec2& uv) {
        EnsureNotFinalized_();
        cpuData_->uvs.push_back(uv);
        cpuData_->needsRepacking = true;
    }

    void Mesh::AddNormal(const glm::vec3& n) {
        EnsureNotFinalized_();
        cpuData_->normals.push_back(n);
        cpuData_->needsRepacking = true;
    }

    void Mesh::AddTangent(const glm::vec3& t) {
        EnsureNotFinalized_();
        cpuData_->tangents.push_back(t);
        cpuData_->needsRepacking = true;
    }

    void Mesh::AddBitangent(const glm::vec3& bt) {
        EnsureNotFinalized_();
        cpuData_->bitangents.push_back(bt);
        cpuData_->needsRepacking = true;
    }

    void Mesh::AddIndex(uint32_t i) {
        EnsureNotFinalized_();
        cpuData_->indices.push_back(i);
        numIndices_ = cpuData_->indices.size();
    }

    void Mesh::CalculateTangentsBitangents_() {
        EnsureNotFinalized_();
        cpuData_->needsRepacking = true;
        cpuData_->tangents.clear();
        cpuData_->bitangents.clear();

        std::vector<uint32_t> indexBuffer;
        const std::vector<uint32_t> * order;
        if (numIndices_ == 0) {
            indexBuffer.resize(numVertices_);
            for (uint32_t i = 0; i < numVertices_; ++i) indexBuffer[i] = i;
            order = &indexBuffer;
        }
        else {
            order = &cpuData_->indices;
        }

        cpuData_->tangents = std::vector<glm::vec3>(numVertices_, glm::vec3(0.0f));
        cpuData_->bitangents = std::vector<glm::vec3>(numVertices_, glm::vec3(0.0f));
        for (int i = 0; i < order->size(); i += 3) {
            const uint32_t i0 = (*order)[i];
            const uint32_t i1 = (*order)[i + 1];
            const uint32_t i2 = (*order)[i + 2];
            auto tanBitan = calculateTangentAndBitangent(cpuData_->vertices[i0], cpuData_->vertices[i1], cpuData_->vertices[i2], cpuData_->uvs[i0], cpuData_->uvs[i1], cpuData_->uvs[i2]);
            
            cpuData_->tangents[i0] += tanBitan.tangent;
            cpuData_->tangents[i1] += tanBitan.tangent;
            cpuData_->tangents[i2] += tanBitan.tangent;

            cpuData_->bitangents[i0] += tanBitan.bitangent;
            cpuData_->bitangents[i1] += tanBitan.bitangent;
            cpuData_->bitangents[i2] += tanBitan.bitangent;
        }

        for (size_t i = 0; i < numVertices_; ++i) {
            glm::vec3 & tangent = cpuData_->tangents[i];
            glm::vec3 & bitangent = cpuData_->bitangents[i];
            const glm::vec3 & normal = cpuData_->normals[i];
            glm::vec3 t = tangent - (normal * glm::dot(normal, tangent));
            t = glm::normalize(t);

            glm::vec3 c = glm::cross(normal, t); // Compute orthogonal 3rd basis vector
            float w = (glm::dot(c, bitangent) < 0) ? -1.0f : 1.0f;
            tangent = t * w;
            bitangent = glm::normalize(c);
        }
    }

    void Mesh::PackCpuData() {
        EnsureNotFinalized_();

        if (cpuData_->tangents.size() == 0 || cpuData_->bitangents.size() == 0) CalculateTangentsBitangents_();

        if (!cpuData_->needsRepacking) return;

        // Pack all data into a single buffer
        cpuData_->data.clear();
        cpuData_->data.resize(numVertices_);
        //_cpuData->data.reserve(_cpuData->vertices.size() * 3 + _cpuData->uvs.size() * 2 + _cpuData->normals.size() * 3 + _cpuData->tangents.size() * 3 + _cpuData->bitangents.size() * 3);
        for (int i = 0; i < numVertices_; ++i) {
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
            GpuMeshData * data = &cpuData_->data[i];
            data->position[0] = cpuData_->vertices[i].x;
            data->position[1] = cpuData_->vertices[i].y;
            data->position[2] = cpuData_->vertices[i].z;
            data->texCoord[0] = cpuData_->uvs[i].x;
            data->texCoord[1] = cpuData_->uvs[i].y;
            data->normal[0] = cpuData_->normals[i].x;
            data->normal[1] = cpuData_->normals[i].y;
            data->normal[2] = cpuData_->normals[i].z;
            data->tangent[0] = cpuData_->tangents[i].x;
            data->tangent[1] = cpuData_->tangents[i].y;
            data->tangent[2] = cpuData_->tangents[i].z;
            data->bitangent[0] = cpuData_->bitangents[i].x;
            data->bitangent[1] = cpuData_->bitangents[i].y;
            data->bitangent[2] = cpuData_->bitangents[i].z;
        }

        dataSizeBytes_ = cpuData_->data.size() * sizeof(GpuMeshData);

        cpuData_->needsRepacking = false;
    }

    // This comes from the Visibility and Occlusion chapter in "Foundations of Game Engine Development, Volume 2: Rendering"
    void Mesh::CalculateAabbs(const glm::mat4& transform) {
        assert(cpuData_->vertices.size() > 0);
        EnsureNotFinalized_();

        // If no indices generate a buffer from [0, num vertices)
        // This does not require CPU data to be repacked
        if (cpuData_->indices.size() == 0) {
            for (uint32_t i = 0; i < numVertices_; ++i) {
                AddIndex(i);
            }
        }

        glm::vec3 vertex = glm::vec3(transform * glm::vec4(cpuData_->vertices[cpuData_->indices[0]], 1.0f));
		glm::vec3 vmin = vertex;
		glm::vec3 vmax = vertex;
        for (size_t i = 0; i < cpuData_->indices.size(); ++i) {
            vertex = glm::vec3(transform * glm::vec4(cpuData_->vertices[cpuData_->indices[i]], 1.0f));
            vmin = glm::min(vmin, vertex);
            vmax = glm::max(vmax, vertex);
        }

        aabb_.vmin = glm::vec4(vmin, 1.0f);
        aabb_.vmax = glm::vec4(vmax, 1.0f);
        //aabb.center = (vmin + vmax) * 0.5f;
        //aabb.size = (vmax - vmin) * 0.5f;
    }

    void Mesh::GenerateLODs() {
        assert(cpuData_->vertices.size() > 0);
        EnsureNotFinalized_();

        // If no indices generate a buffer from [0, num vertices)
        // This does not require CPU data to be repacked
        if (cpuData_->indices.size() == 0) {
            for (uint32_t i = 0; i < numVertices_; ++i) {
                AddIndex(i);
            }
        }

        //meshopt_optimizeVertexCache(cpuData_->indices.data(), cpuData_->indices.data(), cpuData_->indices.size(), cpuData_->vertices.size());

        cpuData_->indicesPerLod.clear();
        cpuData_->indicesPerLod.push_back(cpuData_->indices);
        numIndicesPerLod_.clear();
        numIndicesPerLod_.push_back(cpuData_->indices.size());
        // std::vector<float> errors = {
        //     0.002f, 0.0025f, 0.003f, 0.0035f, 0.004f, 0.0045f, 0.005f
        // };
        const std::vector<float> errors = {
            0.0005f, 0.0005f, 0.001f, 0.001f, 0.005f, 0.005f, 0.005f
        };
        
        const std::vector<float> targetPercentages = {
            0.05f, 0.05f, 0.05f, 0.05f, 0.1f, 0.1f, 0.1f
        };

        for (int i = 0; i < errors.size(); ++i) {
            auto& prevIndices = cpuData_->indicesPerLod[cpuData_->indicesPerLod.size() - 1];
            const size_t targetIndices = size_t(prevIndices.size() * targetPercentages[i]);
            std::vector<uint32_t> simplified(prevIndices.size());
            auto size = meshopt_simplify(simplified.data(), prevIndices.data(), prevIndices.size(), &cpuData_->vertices[0][0], numVertices_, sizeof(float) * 3, targetIndices, errors[i]);
            // If we didn't see at least a 10% reduction, try the more aggressive algorithm
            //if ((prevIndices.size() * 0.9) < double(size)) {
            //   //size = meshopt_simplifySloppy(simplified.data(), prevIndices.data(), prevIndices.size(), &cpuData_->vertices[0][0], numVertices_, sizeof(float) * 3, prevIndices.size() / 2, 0.01f);
            //   error *= 2.0f;
            //   size = meshopt_simplify(simplified.data(), prevIndices.data(), prevIndices.size(), &cpuData_->vertices[0][0], numVertices_, sizeof(float) * 3, prevIndices.size() / 2, error);
            //}
            simplified.resize(size);
            meshopt_optimizeVertexCache(simplified.data(), simplified.data(), size, numVertices_);
            cpuData_->indicesPerLod.push_back(std::move(simplified));
            numIndicesPerLod_.push_back(size);
            if (size < 1024) break;
        }

        meshopt_optimizeVertexCache(cpuData_->indices.data(), cpuData_->indices.data(), cpuData_->indices.size(), cpuData_->vertices.size());
        cpuData_->indicesPerLod[0] = cpuData_->indices;
    }

    size_t Mesh::GetGpuSizeBytes() const {
        EnsureFinalized_();
        return dataSizeBytes_;
    }

    const GpuAABB& Mesh::GetAABB() const {
        EnsureFinalized_();
        return aabb_;
    }

    uint32_t Mesh::GetVertexOffset() const {
        return vertexOffset_;
    }

    uint32_t Mesh::GetIndexOffset(size_t lod) const {
        lod = lod >= indexOffsetPerLod_.size() ? indexOffsetPerLod_.size() - 1 : lod;
        return indexOffsetPerLod_[lod];
    }

    uint32_t Mesh::GetNumIndices(size_t lod) const {
        lod = lod >= numIndicesPerLod_.size() ? numIndicesPerLod_.size() - 1 : lod;
        return numIndicesPerLod_[lod];
    }

    void Mesh::GenerateGpuData_() {
        EnsureNotFinalized_();

        if (cpuData_->indicesPerLod.size() == 0) {
            GenerateLODs();
        }

        vertexOffset_ = GpuMeshAllocator::AllocateVertexData(numVertices_);

        indexOffsetPerLod_.clear();
        for (auto& indices : cpuData_->indicesPerLod) {
            indexOffsetPerLod_.push_back(GpuMeshAllocator::AllocateIndexData(indices.size()));
            // Account for the fact that all vertices are stored in a global GpuBuffer and so
            // the indices need to be offset
            for (size_t i = 0; i < indices.size(); ++i) {
                indices[i] += vertexOffset_;
            }
            
            GpuMeshAllocator::CopyIndexData(indices, indexOffsetPerLod_[indexOffsetPerLod_.size() - 1]);
        }

        GpuMeshAllocator::CopyVertexData(cpuData_->data, vertexOffset_);

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
        delete cpuData_;
        cpuData_ = nullptr;
    }

    void Mesh::FinalizeData() {
        EnsureNotFinalized_();

        PackCpuData();

        if (ApplicationThread::Instance()->CurrentIsApplicationThread()) {
            GenerateGpuData_();
        }
        else {
            ApplicationThread::Instance()->Queue([this]() {
                GenerateGpuData_();
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

        glDrawElementsInstanced(GL_TRIANGLES, numIndices_, GL_UNSIGNED_INT, (const void *)(GetIndexOffset(0) * sizeof(uint32_t)), numInstances);

        additionalBuffers.Unbind();
        //GpuMeshAllocator::UnbindElementArrayBuffer();
    }

    void Mesh::SetFaceCulling(const RenderFaceCulling& cullMode) {
        cullMode_ = cullMode;
    }

    RenderFaceCulling Mesh::GetFaceCulling() const {
        return cullMode_;
    }

    RenderComponent::RenderComponent()
        : meshes(std::make_shared<MeshData>()) {}

    RenderComponent::RenderComponent(const RenderComponent& other) {
        this->meshes = other.meshes;
        this->materials_ = other.materials_;
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
        return materials_.size();
    }

    const std::vector<MaterialPtr>& RenderComponent::GetAllMaterials() const {
        return materials_;
    }

    const MaterialPtr& RenderComponent::GetMaterialAt(size_t index) const {
        return materials_[index];
    }

    void RenderComponent::AddMaterial(MaterialPtr material) {
        materials_.push_back(material);
        MarkChanged();
    }

    void RenderComponent::SetMaterialAt(MaterialPtr material, size_t index) {
        materials_[index] = material;
        MarkChanged();
    }
}