#include "StratusMaterial.h"

namespace stratus {
    Material::Material(const std::string& name, bool registerSelf)
        : _name(name), _registerSelf(registerSelf) {}

    Material::~Material() {}

    // New name must be unique
    void Material::SetName(const std::string& name) {
        std::string old;
        {
            auto ul = _LockWrite();
            old = _name;
            _name = name;
        }
        
        // If the material manager can't change our name then revert it
        if (!MaterialManager::Instance()->NotifyNameChanged(old, shared_from_this())) {
            auto ul = _LockWrite();
            _name = old;
        }
    }

    std::string Material::GetName() const {
        auto sl = _LockRead();
        return _name;
    }

    MaterialPtr Material::CreateSubMaterial() {
        auto ul = _LockWrite();
        auto mat = MaterialPtr(new Material(_name + std::to_string(_subMats.size() + 1), false));
        _subMats.push_back(mat);
        return mat;
    }

    void Material::_Release() {
        auto ul = _LockWrite();
        _registerSelf = false;
    }

    // Get and set material properties
    const glm::vec3& Material::GetDiffuseColor() const {
        auto sl = _LockRead();
        return _diffuseColor;
    }

    const glm::vec3& Material::GetAmbientColor() const {
        auto sl = _LockRead();
        return _ambientColor;
    }

    const glm::vec3& Material::GetBaseReflectivity() const {
        auto sl = _LockRead();
        return _baseReflectivity;
    }

    float Material::GetRoughness() const {
        auto sl = _LockRead();
        return _roughness;
    }

    float Material::GetMetallic() const {
        auto sl = _LockRead();
        return _metallic;
    }

    void Material::SetDiffuseColor(const glm::vec3& diffuse) {
        auto ul = _LockWrite();
        _diffuseColor = diffuse;
    }

    void Material::SetAmbientColor(const glm::vec3& ambient) {
        auto ul = _LockWrite();
        _ambientColor = ambient;
    }

    void Material::SetBaseReflectivity(const glm::vec3& reflectivity) {
        auto ul = _LockWrite();
        _baseReflectivity = reflectivity;
    }

    void Material::SetRoughness(float roughness) {
        auto ul = _LockWrite();
        _roughness = roughness;
    }

    void Material::SetMetallic(float metallic) {
        auto ul = _LockWrite();
        _metallic = metallic;
    }

    // Get and set material properties as textures
    TextureHandle Material::GetDiffuseTexture() const {
        auto sl = _LockRead();
        return _diffuseTexture;
    }

    TextureHandle Material::GetAmbientTexture() const {
        auto sl = _LockRead();
        return _ambientTexture;
    }

    TextureHandle Material::GetNormalMap() const {
        auto sl = _LockRead();
        return _normalMap;
    }

    TextureHandle Material::GetDepthMap() const {
        auto sl = _LockRead();
        return _depthMap;
    }

    TextureHandle Material::GetRoughnessMap() const {
        auto sl = _LockRead();
        return _roughnessMap;
    }

    TextureHandle Material::GetMetallicMap() const {
        auto sl = _LockRead();
        return _metallicMap;
    }

    TextureHandle Material::GetMetallicRoughnessMap() const {
        auto sl = _LockRead();
        return _metallicRoughnessMap;
    }

    void Material::SetDiffuseTexture(TextureHandle handle) {
        auto ul = _LockWrite();
        _diffuseTexture = handle;
    }

    void Material::SetAmbientTexture(TextureHandle handle) {
        auto ul = _LockWrite();
        _ambientTexture = handle;
    }

    void Material::SetNormalMap(TextureHandle handle) {
        auto ul = _LockWrite();
        _normalMap = handle;
    }

    void Material::SetDepthMap(TextureHandle handle) {
        auto ul = _LockWrite();
        _depthMap = handle;
    }

    void Material::SetRoughnessMap(TextureHandle handle) {
        auto ul = _LockWrite();
        _roughnessMap = handle;
    }

    void Material::SetMetallicMap(TextureHandle handle) {
        auto ul = _LockWrite();
        _metallicMap = handle;
    }

    void Material::SetMetallicRoughnessMap(TextureHandle handle) {
        auto ul = _LockWrite();
        _metallicRoughnessMap = handle;
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
        if (!_materials.InsertIfAbsent(std::make_pair(name, mat))) {
            return GetMaterial(name);
        }
        return mat;
    }

    void MaterialManager::ReleaseMaterial(const std::string& name) {
        STRATUS_LOG << "Releasing material: " << name << std::endl;
        auto mat = GetMaterial(name);
        if (mat) {
            mat->_Release();
            _materials.Remove(name);
        }
    }

    MaterialPtr MaterialManager::GetMaterial(const std::string& name) const {
        auto it = _materials.Find(name);
        if (it != _materials.End()) return it->second;
        return nullptr;
    }

    MaterialPtr MaterialManager::GetOrCreateMaterial(const std::string& name) {
        auto mat = GetMaterial(name);
        if (mat) return mat;
        return CreateMaterial(name);
    }

    bool MaterialManager::ContainsMaterial(const std::string& name) const {
        return _materials.Find(name) != _materials.End();
    }

    std::vector<MaterialPtr> MaterialManager::GetAllMaterials() const {
        std::vector<MaterialPtr> mats;
        mats.reserve(_materials.Size());
        for (auto it = _materials.Begin(); it != _materials.End(); ++it) {
            mats.push_back(it->second);
        }
        return mats;
    }

    bool MaterialManager::NotifyNameChanged(const std::string& oldName, MaterialPtr material) {
        bool result = _materials.InsertIfAbsent(std::make_pair(material->GetName(), material));
        if (result) {
            _materials.Remove(oldName);
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
        _materials.Clear();
    }
}