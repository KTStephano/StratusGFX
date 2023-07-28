#pragma once

#include <memory>
#include "StratusCommon.h"
#include "StratusThread.h"
#include "StratusEntity.h"
#include "StratusEntityCommon.h"
#include "StratusRenderComponents.h"
#include "StratusTexture.h"
#include "StratusSystemModule.h"
#include "StratusAsync.h"
#include <vector>
#include <shared_mutex>
#include <unordered_map>

namespace stratus {
    enum class ColorSpace : int {
        NONE,
        SRGB
    };

    enum class TextureLoadingStatus : int {
        LOADING,
        FAILED,
        LOADING_DONE
    };

    SYSTEM_MODULE_CLASS(ResourceManager)
    private:
        struct RawTextureData {
            TextureConfig config;
            TextureHandle handle;
            TextureCoordinateWrapping wrap;
            TextureMinificationFilter min;
            TextureMagnificationFilter mag;
            size_t sizeBytes;
            std::vector<uint8_t *> data;
        };

    public:
        ResourceManager(const ResourceManager&) = delete;
        ResourceManager(ResourceManager&&) = delete;
        ResourceManager& operator=(const ResourceManager&) = delete;
        ResourceManager& operator=(ResourceManager&&) = delete;

        virtual ~ResourceManager();

        Async<Entity> LoadModel(const std::string&, const ColorSpace&, const bool optimizeGraph, RenderFaceCulling defaultCullMode = RenderFaceCulling::CULLING_CCW);
        TextureHandle LoadTexture(const std::string&, const ColorSpace&);
        // prefix is used to select all faces with one string. It ends up expanding to:
        //      prefix + "right." + fileExt
        //      prefix + "left." + fileExt
        //      ...
        //      prefix + "back." + fileExt
        TextureHandle LoadCubeMap(const std::string& prefix, const ColorSpace&, const std::string& fileExt = "jpg");
        Texture LookupTexture(const TextureHandle, TextureLoadingStatus&) const;

        // Default shapes
        EntityPtr CreateCube();
        EntityPtr CreateQuad();

    private:
        // SystemModule inteface
        virtual bool Initialize();
        virtual SystemStatus Update(const double);
        virtual void Shutdown();

    private:
        void ClearAsyncTextureData_();
        void ClearAsyncModelData_();
        void ClearAsyncModelData_(EntityPtr);

    private:
        std::unique_lock<std::shared_mutex> LockWrite_() const { return std::unique_lock<std::shared_mutex>(mutex_); }
        std::shared_lock<std::shared_mutex> LockRead_()  const { return std::shared_lock<std::shared_mutex>(mutex_); }
        EntityPtr LoadModel_(const std::string&, const ColorSpace&, const bool optimizeGraph, RenderFaceCulling);
        // Despite accepting multiple files, it assumes they all have the same format (e.g. for cube texture)
        TextureHandle LoadTextureImpl_(const std::vector<std::string>&, 
                                       const ColorSpace&,
                                       const TextureType type = TextureType::TEXTURE_2D,
                                       const TextureCoordinateWrapping wrap = TextureCoordinateWrapping::REPEAT,
                                       const TextureMinificationFilter min = TextureMinificationFilter::LINEAR_MIPMAP_LINEAR,
                                       const TextureMagnificationFilter mag = TextureMagnificationFilter::LINEAR);
        std::shared_ptr<RawTextureData> LoadTexture_(const std::vector<std::string>&, 
                                                     const TextureHandle, 
                                                     const ColorSpace&,
                                                     const TextureType type = TextureType::TEXTURE_2D,
                                                     const TextureCoordinateWrapping wrap = TextureCoordinateWrapping::REPEAT,
                                                     const TextureMinificationFilter min = TextureMinificationFilter::LINEAR_MIPMAP_LINEAR,
                                                     const TextureMagnificationFilter mag = TextureMagnificationFilter::LINEAR);
        Texture * FinalizeTexture_(const RawTextureData&);

        void InitCube_();
        void InitQuad_();

    private:
        EntityPtr cube_;
        EntityPtr quad_;
        std::unordered_map<std::string, Async<Entity>> loadedModels_;
        std::unordered_map<std::string, Async<Entity>> pendingFinalize_;
        std::unordered_set<MeshPtr> meshFinalizeQueue_;
        std::unordered_set<MeshPtr> generateMeshGpuDataQueue_;
        //std::vector<MeshPtr> _meshFinalizeQueue;
        std::unordered_map<TextureHandle, Async<RawTextureData>> asyncLoadedTextureData_;
        std::unordered_set<TextureHandle> texturesStillLoading_;
        std::unordered_map<TextureHandle, Async<Texture>> loadedTextures_;
        std::unordered_map<std::string, TextureHandle> loadedTexturesByFile_;
        mutable std::shared_mutex mutex_;
    };
}