//
// Created by stephano on 2/17/19.
//

#ifndef STRATUSGFX_RENDERENTITY_H
#define STRATUSGFX_RENDERENTITY_H

#include "StratusCommon.h"
#include "StratusMath.h"
#include <vector>
#include <memory>
#include "StratusGpuBuffer.h"

namespace stratus {
enum class RenderMode {
    ORTHOGRAPHIC,   // 2d - good for menus
    PERSPECTIVE     // 3d
};

enum LightProperties {
    INVISIBLE            = 2,      // material will not be rendered
    FLAT                 = 4,      // material will not interact with light
    DYNAMIC              = 8,      // material fully interacts with all lights
};

enum RenderProperties {
    NONE                 = 0,
    TEXTURED             = 16,     // material has one or more textures
    NORMAL_MAPPED        = 64,     // material has an associated normal map
    HEIGHT_MAPPED        = 128,    // material has an associated depth map
    ROUGHNESS_MAPPED     = 256,    // material has an associated roughness map
    AMBIENT_MAPPED       = 512,    // material has an associated ambient occlusion/environment map
    SHININESS_MAPPED     = 1024,   // material has an associated metalness map
};

/**
 * Each entity should have a set of data that it needs for rendering (vertics, texture coordinates, normals, etc.).
 * In order for instanced rendering to work, the rendering system needs to know which entities have the same
 * A) materials and B) render data.
 */
struct RenderData {
    void * data;
};

enum RenderFaceCulling {
    CULLING_NONE,
    CULLING_CW,     // Clock-wise
    CULLING_CCW,    // Counter-clock-wise
};

/**
 * @see http://devernay.free.fr/cours/opengl/materials.html
 *
 * A material specifies how light will interact with a surface.
 */
struct RenderMaterial {
    glm::vec3 diffuseColor = glm::vec3(1.0f, 0.0f, 0.0f);
    glm::vec3 ambientColor = glm::vec3(1.0f, 0.0f, 0.0f);
    glm::vec3 baseReflectivity = glm::vec3(0.04f);
    float roughness = 0.5f; // (0.0 = smoothest possible, 1.0 = roughest possible)
    float metallic = 0.0f;
    // Not required to have a texture
    TextureHandle texture = -1;
    // Not required to have a normal map
    TextureHandle normalMap = -1;
    // Not required to have a depth map
    TextureHandle depthMap = -1;
    // Not required to have a roughness map
    TextureHandle roughnessMap = -1;
    // Not required to have an ambient map
    TextureHandle ambientMap = -1;
    TextureHandle metalnessMap = -1;
    float heightScale = 0.1;
};

class Mesh {
    friend class Renderer;

    /**
     * Defines certain characteristics such as the texture
     * used, diffuse/specular/ambient colors, as well as the
     * shininess factor.
     */
    RenderMaterial _material;

    /**
     * Encodes key information about _material using an enum.
     */
    RenderProperties _properties = RenderProperties::NONE;

    /**
     * Specific to the underlying API (OpenGL, Vulkan, D3D12)
     */
    RenderData _data;

    struct _MeshData {
        GLuint vao;
        // GLuint vbo;
        // GLuint ebo;
        GpuArrayBuffer buffers;
        uint32_t numVertices;
        uint32_t numIndices;
    };

    _MeshData _drawData;

public:
    Mesh(const std::vector<glm::vec3> & vertices, const std::vector<glm::vec2> & uvs, const std::vector<glm::vec3> & normals);
    Mesh(const std::vector<glm::vec3> & vertices, const std::vector<glm::vec2> & uvs, const std::vector<glm::vec3> & normals, const std::vector<uint32_t> & indices);
    Mesh(const std::vector<glm::vec3> & vertices, const std::vector<glm::vec2> & uvs, const std::vector<glm::vec3> & normals, const std::vector<glm::vec3> & tangents, const std::vector<glm::vec3> & bitangents, const std::vector<uint32_t> & indices);

    Mesh(Mesh &&) = default;
    Mesh(const Mesh &) = delete;

    virtual ~Mesh();

    /**
     * Functions for getting and setting the render material that
     * defines the way light interacts with the surface of this
     * object.
     */
    void setMaterial(const RenderMaterial & material);
    const RenderMaterial & getMaterial() const;
    const RenderProperties & getRenderProperties() const;
    const RenderData & getRenderData() const;

    size_t hashCode() const;
    bool operator==(const Mesh &) const;

    Mesh & operator=(Mesh &&) = delete;
    Mesh & operator=(const Mesh &) = delete;

    /**
     * Determines which (if any) type of face culling should be used.
     */
    RenderFaceCulling cullingMode = CULLING_CW;

protected:
    /**
     * If the rendering system has determined that multiple similar entities can be grouped
     * together, this will be called instead of render(). This call should be nearly identical
     * except it will call the graphics library instanced version of the draw function.
     */
    void render(const int numInstances) const;

    /**
     * Binds any data buffers for the next call to render()
     */
    void bind() const;

    /**
     * Unbind all data buffers associated with rendering
     */
    void unbind() const;

private:
    void _setProperties(uint32_t properties);
    void _enableProperties(uint32_t properties);
    void _disableProperties(uint32_t properties);
};
}

namespace std {
    template<>
    struct hash<stratus::Mesh> {
        size_t operator()(const stratus::Mesh & m) const {
            return m.hashCode();
        }
    };
}

namespace stratus {
class RenderEntity {
    /**
     * This is used by the renderer to decide which shader
     * program to use.
     */
    LightProperties _properties;

public:
    glm::vec3 position = glm::vec3(0.0f);
    glm::vec3 scale = glm::vec3(1.0f);
    Rotation rotation;
    glm::mat4 model = glm::mat4(1.0f);
    std::vector<std::shared_ptr<Mesh>> meshes;
    std::vector<RenderEntity> nodes;

    /**
     * @param properties render properties which decides which
     *      shader to use
     */
    RenderEntity(LightProperties properties = LightProperties::DYNAMIC);
    RenderEntity(const RenderEntity & other) = default;
    RenderEntity(RenderEntity && other) = default;
    RenderEntity & operator=(const RenderEntity & other) = default;
    RenderEntity & operator=(RenderEntity && other) = default;
    virtual ~RenderEntity();

    void setLightProperties(const LightProperties & properties);
    const LightProperties & getLightProperties() const;

    size_t hashCode() const;
    bool operator==(const RenderEntity &) const;
};
}

namespace std {
    template<>
    struct hash<stratus::RenderEntity> {
        size_t operator()(const stratus::RenderEntity & e) const {
            return e.hashCode();
        }
    };
}

#endif //STRATUSGFX_RENDERENTITY_H
