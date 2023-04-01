#pragma once

#include "StratusCommon.h"
#include "StratusTexture.h"
#include "StratusConcurrentHashMap.h"
#include <vector>
#include <shared_mutex>
#include <string>
#include <memory>
#include "StratusLog.h"
#include "StratusSystemModule.h"

namespace stratus {
    class Material;
    typedef std::shared_ptr<Material> MaterialPtr;

    /**
     * @see http://devernay.free.fr/cours/opengl/materials.html
     *
     * A material specifies how light will interact with a surface.
     */
    class Material : public std::enable_shared_from_this<Material> {
        friend class MaterialManager;

        // Only material manager should create
        Material(const std::string& name, bool registerSelf);

    public:
        ~Material();

        Material(const Material&) = delete;
        Material(Material&&) = delete;
        Material& operator=(const Material&) = delete;
        Material& operator=(Material&&) = delete;

        // New name must be unique
        void SetName(const std::string&);
        std::string GetName() const;

        // Creates an un-named sub material (only parent is registered with MaterialManager)
        MaterialPtr CreateSubMaterial();

        // Helper comparison functions
        bool operator==(const Material& other) const { return GetName() == other.GetName(); }
        bool operator!=(const Material& other) const { return !(*this == other); }

        // Get and set material properties
        const glm::vec4& GetDiffuseColor() const;
        const glm::vec3& GetAmbientColor() const;
        const glm::vec3& GetBaseReflectivity() const;
        float GetRoughness() const;
        float GetMetallic() const;

        void SetDiffuseColor(const glm::vec4&);
        void SetAmbientColor(const glm::vec3&);
        void SetBaseReflectivity(const glm::vec3&);
        void SetRoughness(float);
        void SetMetallic(float);

        // Get and set material properties as textures
        TextureHandle GetDiffuseTexture() const;
        TextureHandle GetAmbientTexture() const;
        TextureHandle GetNormalMap() const;
        TextureHandle GetDepthMap() const;
        TextureHandle GetRoughnessMap() const;
        TextureHandle GetMetallicMap() const;
        TextureHandle GetMetallicRoughnessMap() const;

        void SetDiffuseTexture(TextureHandle);
        void SetAmbientTexture(TextureHandle);
        void SetNormalMap(TextureHandle);
        void SetDepthMap(TextureHandle);
        void SetRoughnessMap(TextureHandle);
        void SetMetallicMap(TextureHandle);
        // Things like GLTF 2.0 permit a combined metallic-roughness map
        void SetMetallicRoughnessMap(TextureHandle);

        void MarkChanged();
        bool ChangedWithinLastFrame();

    private:
        //std::unique_lock<std::shared_mutex> _LockWrite() const { return std::unique_lock<std::shared_mutex>(_mutex); }
        //std::shared_lock<std::shared_mutex> _LockRead()  const { return std::shared_lock<std::shared_mutex>(_mutex); }
        // TODO: We will be accessing material state too often to have to lock every time. Ensure thread safety similar
        // to how Entities are handled where each system updates one at a time and can parallelize themselves while ensuring
        // no material is changed by multiple threads at once.
        int LockWrite_() const { return 0; }
        int LockRead_()  const { return 0; }

        void Release_();
    
    private:
        //mutable std::shared_mutex _mutex;
        std::string name_;
        // Register self with material manager
        bool registerSelf_;
        uint64_t lastFrameChanged_ = 0;
        glm::vec4 diffuseColor_ = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
        glm::vec3 ambientColor_ = glm::vec3(1.0f, 0.0f, 0.0f);
        glm::vec3 baseReflectivity_ = glm::vec3(0.05f);
        float roughness_ = 0.5f; // (0.0 = smoothest possible, 1.0 = roughest possible)
        float metallic_ = 0.04f; // 0.04 is good for many non-metallic surfaces
        // Not required to have a texture
        TextureHandle diffuseTexture_ = TextureHandle::Null();
        // Not required to have an ambient texture
        TextureHandle ambientTexture_ = TextureHandle::Null();
        // Not required to have a normal map
        TextureHandle normalMap_ = TextureHandle::Null();
        // Not required to have a depth map
        TextureHandle depthMap_ = TextureHandle::Null();
        // Not required to have a roughness map
        TextureHandle roughnessMap_ = TextureHandle::Null();
        // Not required to have a metallic map
        TextureHandle metallicMap_ = TextureHandle::Null();
        // Not required to have a metallic-roughness map
        TextureHandle metallicRoughnessMap_ = TextureHandle::Null();
        std::vector<MaterialPtr> subMats_;
    };

    SYSTEM_MODULE_CLASS(MaterialManager)
        virtual ~MaterialManager();

        // Creating and querying materials
        MaterialPtr CreateMaterial(const std::string& name);
        // Removes material from manager's cache - once the last outstanding pointer to it
        // is dropped the material will go out of scope
        void ReleaseMaterial(const std::string& name);
        MaterialPtr GetMaterial(const std::string& name) const;
        MaterialPtr GetOrCreateMaterial(const std::string& name);
        bool ContainsMaterial(const std::string& name) const;
        std::vector<MaterialPtr> GetAllMaterials() const;

        bool NotifyNameChanged(const std::string& oldName, MaterialPtr);

        MaterialPtr CreateDefault();

    private:
        // SystemModule inteface
        virtual bool Initialize();
        virtual SystemStatus Update(const double);
        virtual void Shutdown();

    private:
        ConcurrentHashMap<std::string, MaterialPtr> materials_;
    };
}