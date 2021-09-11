#pragma once

#include <vector>
#include <memory>
#include "StratusCommon.h"
#include "StratusGpuBuffer.h"
#include "StratusMaterial.h"
#include "StratusMath.h"

namespace stratus {
    enum class RenderFaceCulling : int {
        CULLING_NONE,
        CULLING_CW,     // Clock-wise
        CULLING_CCW,    // Counter-clock-wise
    };

    struct RenderMesh;
    struct RenderNode;
    struct Entity;

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

        void CalculateTangentsBitangents() const;
        void GenerateCpuData() const;
        void GenerateGpuData() const;
        const GpuArrayBuffer& GetData() const;

        void Render(size_t numInstances, const GpuArrayBuffer& additionalBuffers) const;

    private:
        mutable GpuArrayBuffer _buffers;
        mutable std::vector<glm::vec3> _vertices;
        mutable std::vector<glm::vec2> _uvs;
        mutable std::vector<glm::vec3> _normals;
        mutable std::vector<glm::vec3> _tangents;
        mutable std::vector<glm::vec3> _bitangents;
        mutable std::vector<uint32_t> _indices;
        mutable std::vector<float> _data;
        mutable uint32_t _numVertices;
        mutable uint32_t _numIndices;
        mutable bool _isCpuDirty = true;
        mutable bool _isGpuDirty = true;
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

        const glm::vec3& GetWorldPosition() const;

        // This will be entity world transform * node transform
        const glm::mat4& GetWorldTransform() const;

        // True by default
        void EnableLightInteraction(bool enabled);
        void SetInvisible(bool invisible);
        void SetFaceCullMode(const RenderFaceCulling&);

        bool GetLightInteractionEnabled() const;
        bool GetInvisible() const;
        RenderFaceCulling GetFaceCullMode() const;

        bool operator==(const RenderNode& other) const;
        bool operator!=(const RenderNode& other) const { return !(*this == other); }

        size_t HashCode() const;

        const std::shared_ptr<Entity>& GetOwner() const;
        void SetOwner(const std::shared_ptr<Entity>&);

    private:
        std::vector<RenderMeshContainer> _meshes;
        mutable bool _transformIsDirty = true;
        glm::vec3 _position = glm::vec3(0.0f);
        mutable glm::vec3 _worldPosition = glm::vec3(0.0f);
        Rotation _rotation;
        glm::vec3 _scale = glm::vec3(1.0f);
        mutable glm::mat4 _worldTransform = glm::mat4(1.0f);
        glm::mat4 _worldEntityTransform = glm::mat4(1.0f);
        bool _lightInteractionEnabled = true;
        bool _invisible = false;
        RenderFaceCulling _cullMode;
        std::shared_ptr<Entity> _owner;
    };

    // This makes it easy to insert a render node into a hash set/map
    struct RenderNodeView {
        RenderNodeView() {}
        RenderNodeView(const RenderNodePtr& node) : _node(node) {}
        size_t HashCode() const { return _node->HashCode(); }
        const RenderNodePtr& Get() const { return _node; }

        bool operator==(const RenderNodeView& other) const {
            return *_node == *other._node;
        }

        bool operator!=(const RenderNodeView& other) const {
            return !(*this == other);
        }

    private:
        RenderNodePtr _node;
    };
}

namespace std {
    template<>
    struct hash<stratus::RenderNodeView> {
        size_t operator()(const stratus::RenderNodeView & v) const {
            return v.HashCode();
        }
    };
}