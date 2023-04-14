#include "StratusMaterial.h"
#include "StratusEngine.h"

namespace stratus {
    Material::Material(const std::string& name, bool registerSelf)
        : name_(name), registerSelf_(registerSelf) {}

    Material::~Material() {}

    void Material::MarkChanged() {
        auto ul = LockWrite_();
        lastFrameChanged_ = INSTANCE(Engine)->FrameCount();
    }

    bool Material::ChangedWithinLastFrame() {
        auto sl = LockRead_();
        auto diff = INSTANCE(Engine)->FrameCount() - lastFrameChanged_;
        return diff <= 1;
    }

    // New name must be unique
    void Material::SetName(const std::string& name) {
        std::string old;
        {
            auto ul = LockWrite_();
            old = name_;
            name_ = name;
        }
        
        // If the material manager can't change our name then revert it
        if (!MaterialManager::Instance()->NotifyNameChanged(old, shared_from_this())) {
            MarkChanged();
            auto ul = LockWrite_();
            name_ = old;
        }
    }

    std::string Material::GetName() const {
        auto sl = LockRead_();
        return name_;
    }

    MaterialPtr Material::CreateSubMaterial() {
        auto ul = LockWrite_();
        auto mat = MaterialPtr(new Material(name_ + std::to_string(subMats_.size() + 1), false));
        subMats_.push_back(mat);
        return mat;
    }

    void Material::Release_() {
        auto ul = LockWrite_();
        registerSelf_ = false;
    }

    // Get and set material properties
    const glm::vec4& Material::GetDiffuseColor() const {
        auto sl = LockRead_();
        return diffuseColor_;
    }

    const glm::vec3& Material::GetEmissiveColor() const {
        auto sl = LockRead_();
        return emissiveColor;
    }

    const glm::vec3& Material::GetBaseReflectivity() const {
        auto sl = LockRead_();
        return baseReflectivity_;
    }

    float Material::GetRoughness() const {
        auto sl = LockRead_();
        return roughness_;
    }

    float Material::GetMetallic() const {
        auto sl = LockRead_();
        return metallic_;
    }

    void Material::SetDiffuseColor(const glm::vec4& diffuse) {
        MarkChanged();
        auto ul = LockWrite_();
        diffuseColor_ = diffuse;
    }

    void Material::SetEmissiveColor(const glm::vec3& ambient) {
        MarkChanged();
        auto ul = LockWrite_();
        emissiveColor = ambient;
    }

    void Material::SetBaseReflectivity(const glm::vec3& reflectivity) {
        MarkChanged();
        auto ul = LockWrite_();
        baseReflectivity_ = reflectivity;
    }

    void Material::SetRoughness(float roughness) {
        MarkChanged();
        auto ul = LockWrite_();
        roughness_ = roughness;
    }

    void Material::SetMetallic(float metallic) {
        MarkChanged();
        auto ul = LockWrite_();
        metallic_ = metallic;
    }

    // Get and set material properties as textures
    TextureHandle Material::GetDiffuseTexture() const {
        auto sl = LockRead_();
        return diffuseTexture_;
    }

    TextureHandle Material::GetEmissiveTexture() const {
        auto sl = LockRead_();
        return emissiveTexture_;
    }

    TextureHandle Material::GetNormalMap() const {
        auto sl = LockRead_();
        return normalMap_;
    }

    TextureHandle Material::GetDepthMap() const {
        auto sl = LockRead_();
        return depthMap_;
    }

    TextureHandle Material::GetRoughnessMap() const {
        auto sl = LockRead_();
        return roughnessMap_;
    }

    TextureHandle Material::GetMetallicMap() const {
        auto sl = LockRead_();
        return metallicMap_;
    }

    TextureHandle Material::GetMetallicRoughnessMap() const {
        auto sl = LockRead_();
        return metallicRoughnessMap_;
    }

    void Material::SetDiffuseTexture(TextureHandle handle) {
        MarkChanged();
        auto ul = LockWrite_();
        diffuseTexture_ = handle;
    }

    void Material::SetAmbientTexture(TextureHandle handle) {
        MarkChanged();
        auto ul = LockWrite_();
        emissiveTexture_ = handle;
    }

    void Material::SetNormalMap(TextureHandle handle) {
        MarkChanged();
        auto ul = LockWrite_();
        normalMap_ = handle;
    }

    void Material::SetDepthMap(TextureHandle handle) {
        MarkChanged();
        auto ul = LockWrite_();
        depthMap_ = handle;
    }

    void Material::SetRoughnessMap(TextureHandle handle) {
        MarkChanged();
        auto ul = LockWrite_();
        roughnessMap_ = handle;
    }

    void Material::SetMetallicMap(TextureHandle handle) {
        MarkChanged();
        auto ul = LockWrite_();
        metallicMap_ = handle;
    }

    void Material::SetMetallicRoughnessMap(TextureHandle handle) {
        MarkChanged();
        auto ul = LockWrite_();
        metallicRoughnessMap_ = handle;
    }

    MaterialManager::MaterialManager() {}

    MaterialManager::~MaterialManager() {}

    MaterialPtr MaterialManager::CreateMaterial(const std::string& name) {
        STRATUS_LOG << "Attempting to create material: " << name << std::endl;

        // Check if we already have it
        MaterialPtr mat = GetMaterial(name);
        if (mat != nullptr) return mat;

        mat = MaterialPtr(new Material(name, true));
        // If we fail to insert, return the one we have
        if (!materials_.InsertIfAbsent(std::make_pair(name, mat))) {
            return GetMaterial(name);
        }
        return mat;
    }

    void MaterialManager::ReleaseMaterial(const std::string& name) {
        STRATUS_LOG << "Releasing material: " << name << std::endl;
        auto mat = GetMaterial(name);
        if (mat) {
            mat->Release_();
            materials_.Remove(name);
        }
    }

    MaterialPtr MaterialManager::GetMaterial(const std::string& name) const {
        auto it = materials_.Find(name);
        if (it != materials_.End()) return it->second;
        return nullptr;
    }

    MaterialPtr MaterialManager::GetOrCreateMaterial(const std::string& name) {
        auto mat = GetMaterial(name);
        if (mat) return mat;
        return CreateMaterial(name);
    }

    bool MaterialManager::ContainsMaterial(const std::string& name) const {
        return materials_.Find(name) != materials_.End();
    }

    std::vector<MaterialPtr> MaterialManager::GetAllMaterials() const {
        std::vector<MaterialPtr> mats;
        mats.reserve(materials_.Size());
        for (auto it = materials_.Begin(); it != materials_.End(); ++it) {
            mats.push_back(it->second);
        }
        return mats;
    }

    bool MaterialManager::NotifyNameChanged(const std::string& oldName, MaterialPtr material) {
        bool result = materials_.InsertIfAbsent(std::make_pair(material->GetName(), material));
        if (result) {
            materials_.Remove(oldName);
        }        
        return result;
    }

    MaterialPtr MaterialManager::CreateDefault() {
        return MaterialPtr(new Material("Default", false));
    }

    bool MaterialManager::Initialize() {
        return true;
    }

    SystemStatus MaterialManager::Update(const double) {
        return SystemStatus::SYSTEM_CONTINUE;
    }

    void MaterialManager::Shutdown() {
        materials_.Clear();
    }
}