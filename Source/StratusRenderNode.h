#pragma once

#include <vector>
#include <memory>
#include "StratusCommon.h"
#include "StratusGpuBuffer.h"
#include "StratusMaterial.h"
#include "StratusMath.h"

namespace stratus {
    enum class _RenderFaceCulling {
        CULLING_NONE,
        CULLING_CW,     // Clock-wise
        CULLING_CCW,    // Counter-clock-wise
    };

    struct RenderMesh;
    struct RenderNode;

    typedef std::shared_ptr<RenderMesh> RenderMeshPtr;
    typedef std::shared_ptr<RenderNode> RenderNodePtr;

    struct RenderMesh {
        ~RenderMesh() = default;
    
        void AddVertex(const glm::vec3&);
        void AddUV(const glm::vec2&);
        void AddNormal(const glm::vec3&);
        void AddTangent(const glm::vec3&);
        void AddBitangent(const glm::vec3&);
        void AddIndex(uint32_t);

        void CalculateTangentsBitangents();
        void GenerateGpuData();
        const GpuArrayBuffer& GetData() const;

        void Render(size_t numInstances, const GpuArrayBuffer& additionalBuffers);

    private:
        GpuArrayBuffer _buffers;
        std::vector<glm::vec3> _vertices;
        std::vector<glm::vec2> _uvs;
        std::vector<glm::vec3> _normals;
        std::vector<glm::vec3> _tangents;
        std::vector<glm::vec3> _bitangents;
        std::vector<uint32_t> _indices;
        uint32_t _numVertices;
        uint32_t _numIndices;
    };

    struct RenderMeshContainer {
        RenderMeshPtr mesh;
        MaterialPtr material;
    };

    // A render node contains one or more material+mesh combinations as well as
    // a local transform. A render node itself is meant to be attached to an Entity
    // and the full world transform is derived by doing Entity->Transform * Node->Transform.
    struct RenderNode {
        // Deep copy of transform data - shallow copy of render data
        RenderNodePtr Copy() const;

        size_t GetNumMeshContainers() const;
        const RenderMeshContainer * GetMeshContainer(size_t index) const;
        void AddMeshContainer(const RenderMeshContainer&);

        // Sets the material for all mesh containers
        void SetMaterial(const MaterialPtr&);
        // Sets material for individual container
        void SetMaterialFor(size_t containerIndex, const MaterialPtr&);

        // Transform info
        void SetLocalPosition(const glm::vec3&);
        void SetLocalRotation(const Rotation&);
        void SetLocalScale(const glm::vec3&);
        void SetLocalPosRotScale(const glm::vec3&, const Rotation&, const glm::vec3&);
        // Called by Entity when its transform info changes
        void SetWorldTransform(const glm::mat4&);

        const glm::vec3& GetLocalPosition() const;
        const Rotation& GetLocalRotation() const;
        const glm::vec3& GetLocalScale() const;

        // This will be entity world transform * node transform
        const glm::mat4& GetWorldTransform() const;

        // True by default
        void EnableLightInteraction(bool enabled);
        void SetFaceCullMode(const _RenderFaceCulling&);

        bool GetLightInteractionEnabled() const;
        _RenderFaceCulling GetFaceCullMode() const;

        bool operator==(const RenderNode& other) const;
        bool operator!=(const RenderNode& other) const { return !(*this == other); }

    private:
        std::vector<RenderMeshContainer> _meshes;
        mutable bool _transformIsDirty = true;
        glm::vec3 _position = glm::vec3(0.0f);
        Rotation _rotation;
        glm::vec3 _scale = glm::vec3(1.0f);
        mutable glm::mat4 _worldTransform = glm::mat4(1.0f);
        glm::mat4 _worldEntityTransform = glm::mat4(1.0f);
        bool _lightInteractionEnabled = true;
        _RenderFaceCulling _cullMode;
    };
}