#include "StratusRenderNode.h"
#include "StratusUtils.h"
#include "StratusEntity.h"
#include "StratusApplicationThread.h"

namespace stratus {
    void RenderMesh::AddVertex(const glm::vec3& v) {
        _vertices.push_back(v);
        _numVertices = _vertices.size();
        _isCpuDirty = true;
        _isGpuDirty = true;
        _isGpuDataDirty = true;
    }

    void RenderMesh::AddUV(const glm::vec2& uv) {
        _uvs.push_back(uv);
        _isCpuDirty = true;
        _isGpuDirty = true;
        _isGpuDataDirty = true;
    }

    void RenderMesh::AddNormal(const glm::vec3& n) {
        _normals.push_back(n);
        _isCpuDirty = true;
        _isGpuDirty = true;
        _isGpuDataDirty = true;
    }

    void RenderMesh::AddTangent(const glm::vec3& t) {
        _tangents.push_back(t);
        _isCpuDirty = true;
        _isGpuDirty = true;
        _isGpuDataDirty = true;
    }

    void RenderMesh::AddBitangent(const glm::vec3& bt) {
        _bitangents.push_back(bt);
        _isCpuDirty = true;
        _isGpuDirty = true;
        _isGpuDataDirty = true;
    }

    void RenderMesh::AddIndex(uint32_t i) {
        _indices.push_back(i);
        _numIndices = _indices.size();
        _isCpuDirty = true;
        _isGpuDirty = true;
        _isGpuDataDirty = true;
    }

    void RenderMesh::CalculateTangentsBitangents() const {
        _tangents.clear();
        _bitangents.clear();

        std::vector<uint32_t> indexBuffer;
        const std::vector<uint32_t> * order;
        if (_numIndices == 0) {
            indexBuffer.resize(_numVertices);
            for (uint32_t i = 0; i < _numVertices; ++i) indexBuffer[i] = i;
            order = &indexBuffer;
        }
        else {
            order = &_indices;
        }

        _tangents = std::vector<glm::vec3>(_numVertices, glm::vec3(0.0f));
        _bitangents = std::vector<glm::vec3>(_numVertices, glm::vec3(0.0f));
        for (int i = 0; i < order->size(); i += 3) {
            const uint32_t i0 = (*order)[i];
            const uint32_t i1 = (*order)[i + 1];
            const uint32_t i2 = (*order)[i + 2];
            auto tanBitan = calculateTangentAndBitangent(_vertices[i0], _vertices[i1], _vertices[i2], _uvs[i0], _uvs[i1], _uvs[i2]);
            
            _tangents[i0] += tanBitan.tangent;
            _tangents[i1] += tanBitan.tangent;
            _tangents[i2] += tanBitan.tangent;

            _bitangents[i0] += tanBitan.bitangent;
            _bitangents[i1] += tanBitan.bitangent;
            _bitangents[i2] += tanBitan.bitangent;
        }

        for (size_t i = 0; i < _numVertices; ++i) {
            glm::vec3 & tangent = _tangents[i];
            glm::vec3 & bitangent = _bitangents[i];
            const glm::vec3 & normal = _normals[i];
            glm::vec3 t = tangent - (normal * glm::dot(normal, tangent));
            t = glm::normalize(t);

            glm::vec3 c = glm::cross(normal, t); // Compute orthogonal 3rd basis vector
            float w = (glm::dot(c, bitangent) < 0) ? -1.0f : 1.0f;
            tangent = t * w;
            bitangent = glm::normalize(c);
        }
    }

    void RenderMesh::GenerateCpuData() const {
        if (!_isCpuDirty) return;
        _isCpuDirty = false;

        if (_tangents.size() == 0 || _bitangents.size() == 0) CalculateTangentsBitangents();

        // Pack all data into a single buffer
        _data.clear();
        _data.reserve(_vertices.size() * 3 + _uvs.size() * 2 + _normals.size() * 3 + _tangents.size() * 3 + _bitangents.size() * 3);
        for (int i = 0; i < _numVertices; ++i) {
            _data.push_back(_vertices[i].x);
            _data.push_back(_vertices[i].y);
            _data.push_back(_vertices[i].z);
            _data.push_back(_uvs[i].x);
            _data.push_back(_uvs[i].y);
            _data.push_back(_normals[i].x);
            _data.push_back(_normals[i].y);
            _data.push_back(_normals[i].z);
            _data.push_back(_tangents[i].x);
            _data.push_back(_tangents[i].y);
            _data.push_back(_tangents[i].z);
            _data.push_back(_bitangents[i].x);
            _data.push_back(_bitangents[i].y);
            _data.push_back(_bitangents[i].z);
        }

        _dataSizeBytes = _data.size() * sizeof(float);

        // Clear out existing buffers to conserve system memory
        _vertices.clear();
        _uvs.clear();
        _normals.clear();
        _tangents.clear();
        _bitangents.clear();
    }

    size_t RenderMesh::GetGpuSizeBytes() const {
        return _dataSizeBytes;
    }

    void RenderMesh::GenerateGpuData() const {
        if (!_isGpuDirty) return;

        _buffers = GpuArrayBuffer();
        GpuBuffer buffer;
        if (_numIndices > 0) {
            buffer = GpuBuffer(GpuBufferType::INDEX_BUFFER, nullptr, _indices.size() * sizeof(uint32_t));
            _buffers.AddBuffer(buffer);
            _indicesMapped = buffer.MapReadWrite();
        }

        // To get to the next full element we have to skip past a set of vertices (3), uvs (2), normals (3), tangents (3), and bitangents (3)
        buffer = GpuBuffer(GpuBufferType::PRIMITIVE_BUFFER, nullptr, _data.size() * sizeof(float));
        _buffers.AddBuffer(buffer);
        _primitiveMapped = buffer.MapReadWrite();
        
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

        _isGpuDirty = false;
    }

    void RenderMesh::FinalizeGpuData() const {
        if (!_isGpuDataDirty) return;

        if (_numIndices > 0) {
            memcpy(_indicesMapped, (const void *)_indices.data(), _indices.size() * sizeof(uint32_t));

        }

        memcpy(_primitiveMapped, (const void *)_data.data(), _data.size() * sizeof(float));

        // Clear out existing buffers to conserve system memory
        _indices.clear();
        _data.clear();

        _isGpuDataDirty = false;

        if (ApplicationThread::Instance()->CurrentIsApplicationThread()) {
            UnmapAllGpuBuffers();
        }
        else {
            ApplicationThread::Instance()->Queue([this]() { UnmapAllGpuBuffers(); });
        }
    }

    void RenderMesh::UnmapAllGpuBuffers() const {
        _buffers.UnmapAllReadWrite();
    }

    bool RenderMesh::IsCpuDirty() const {
        return _isCpuDirty;
    }

    bool RenderMesh::IsGpuDirty() const {
        return _isGpuDirty;
    }

    bool RenderMesh::IsGpuDataDirty() const {
        return _isGpuDataDirty;
    }

    bool RenderMesh::Complete() const {
        return !_isCpuDirty && !_isGpuDirty && !_isGpuDataDirty;
    }

    const GpuArrayBuffer& RenderMesh::GetData() const {
        return _buffers;
    }

    void RenderMesh::Render(size_t numInstances, const GpuArrayBuffer& additionalBuffers) const {
        // GenerateCpuData();
        // GenerateGpuData();
        // FinalizeGpuData();

        // We don't want to run if our memory is mapped since another thread might be
        // writing data to a buffer's memory region
        if (!Complete() || _buffers.IsMemoryMapped()) return;

        if (_primitiveMapped != nullptr || _indicesMapped != nullptr) {
            _buffers.UnmapAllReadWrite();
            _primitiveMapped = nullptr;
            _indicesMapped = nullptr;
        }

        _buffers.Bind();
        additionalBuffers.Bind();

        if (_numIndices > 0) {
            glDrawElementsInstanced(GL_TRIANGLES, _numIndices, GL_UNSIGNED_INT, (void *)0, numInstances);
        }
        else {
            glDrawArraysInstanced(GL_TRIANGLES, 0, _numVertices, numInstances);
        }

        additionalBuffers.Unbind();
        _buffers.Unbind();
    }

    void RenderMesh::SetFaceCulling(const RenderFaceCulling& cullMode) {
        _cullMode = cullMode;
    }

    RenderFaceCulling RenderMesh::GetFaceCulling() const {
        return _cullMode;
    }

    RenderNodePtr RenderNode::Copy() const {
        return std::make_shared<RenderNode>(*this);
    }

    size_t RenderNode::GetNumMeshContainers() const {
        return _meshes.size();
    }

    const RenderMeshContainer * RenderNode::GetMeshContainer(size_t index) const {
        return &_meshes[index];
    }

    void RenderNode::AddMeshContainer(const RenderMeshContainer& mesh) {
        _meshes.push_back(mesh);
    }

    void RenderNode::SetMaterial(const MaterialPtr& mat) {
        for (size_t i = 0; i < GetNumMeshContainers(); ++i) {
            SetMaterialFor(i, mat);
        }
    }

    void RenderNode::SetMaterialFor(size_t containerIndex, const MaterialPtr& mat) {
        _meshes[containerIndex].material = mat;
    }

    void RenderNode::SetLocalPosition(const glm::vec3& pos) {
        _position = pos;
        _transformIsDirty = true;
    }

    void RenderNode::SetLocalRotation(const Rotation& rot) {
        _rotation = rot;
        _transformIsDirty = true;
    }

    void RenderNode::SetLocalScale(const glm::vec3& scale) {
        _scale = scale;
        _transformIsDirty = true;
    }

    void RenderNode::SetLocalPosRotScale(const glm::vec3& pos, const Rotation& rot, const glm::vec3& scale) {
        SetLocalPosition(pos);
        SetLocalRotation(rot);
        SetLocalScale(scale);
    }

    void RenderNode::SetWorldTransform(const glm::mat4& mat) {
        _worldEntityTransform = mat;
        _transformIsDirty = true;
    }

    const glm::vec3& RenderNode::GetLocalPosition() const {
        return _position;
    }

    const Rotation& RenderNode::GetLocalRotation() const {
        return _rotation;
    }

    const glm::vec3& RenderNode::GetLocalScale() const {
        return _scale;
    }

    const glm::vec3& RenderNode::GetWorldPosition() const {
        if (_transformIsDirty) {
            auto& mat = GetWorldTransform();
        }
        return _worldPosition;
    }

    const glm::mat4& RenderNode::GetWorldTransform() const {
        if (_transformIsDirty) {
            _worldTransform = constructTransformMat(_rotation, _position, _scale);
            _worldTransform = _worldEntityTransform * _worldTransform;
            _worldPosition = glm::vec3(_worldTransform * glm::vec4(_position, 1.0f));
            _transformIsDirty = false;
        }
        return _worldTransform;
    }

    void RenderNode::EnableLightInteraction(bool enabled) {
        _lightInteractionEnabled = enabled;
    }

    bool RenderNode::GetLightInteractionEnabled() const {
        return _lightInteractionEnabled;
    }

    void RenderNode::SetInvisible(bool invisible) {
        _invisible = invisible;
    }

    bool RenderNode::GetInvisible() const {
        return _invisible;
    }

    bool RenderNode::operator==(const RenderNode& other) const {
        if (GetNumMeshContainers() != other.GetNumMeshContainers()) return false;
        // Lit entities have a separate pipeline from non-lit entities
        if (_lightInteractionEnabled != other._lightInteractionEnabled) return false;
        for (size_t i = 0; i < GetNumMeshContainers(); ++i) {
            if ((GetMeshContainer(i)->mesh != other.GetMeshContainer(i)->mesh) ||
                (GetMeshContainer(i)->material != other.GetMeshContainer(i)->material)) {
                return false;
            }
        }
        return true;
    }

    size_t RenderNode::HashCode() const {
        size_t code = 0;
        for (auto& mesh : _meshes) {
            code += std::hash<RenderMeshPtr>{}(mesh.mesh);
            code += std::hash<MaterialPtr>{}(mesh.material);
        }

        code += std::hash<bool>{}(_lightInteractionEnabled);

        return code;
    }

    const std::shared_ptr<Entity>& RenderNode::GetOwner() const {
        return _owner;
    }

    void RenderNode::SetOwner(const std::shared_ptr<Entity>& owner) {
        _owner = owner;
    }
}