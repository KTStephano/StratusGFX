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

        Async<Entity> LoadModel(const std::string&, RenderFaceCulling defaultCullMode = RenderFaceCulling::CULLING_CCW);
        TextureHandle LoadTexture(const std::string&, const bool srgb);
        // prefix is used to select all faces with one string. It ends up expanding to:
        //      prefix + "right." + fileExt
        //      prefix + "left." + fileExt
        //      ...
        //      prefix + "back." + fileExt
        TextureHandle LoadCubeMap(const std::string& prefix, const bool srgb, const std::string& fileExt = "jpg");
        bool GetTexture(const TextureHandle, Async<Texture>&) const;
        Async<Texture> LookupTexture(TextureHandle handle) const;

        // Default shapes
        EntityPtr CreateCube();
        EntityPtr CreateQuad();

    private:
        // SystemModule inteface
        virtual bool Initialize();
        virtual SystemStatus Update(const double);
        virtual void Shutdown();

    private:
        void _ClearAsyncTextureData();
        void _ClearAsyncModelData();
        void _ClearAsyncModelData(EntityPtr);

    private:
        std::unique_lock<std::shared_mutex> _LockWrite() const { return std::unique_lock<std::shared_mutex>(_mutex); }
        std::shared_lock<std::shared_mutex> _LockRead()  const { return std::shared_lock<std::shared_mutex>(_mutex); }
        EntityPtr _LoadModel(const std::string&, RenderFaceCulling);
        // Despite accepting multiple files, it assumes they all have the same format (e.g. for cube texture)
        TextureHandle _LoadTextureImpl(const std::vector<std::string>&, 
                                       const bool srgb,
                                       const TextureType type = TextureType::TEXTURE_2D,
                                       const TextureCoordinateWrapping wrap = TextureCoordinateWrapping::REPEAT,
                                       const TextureMinificationFilter min = TextureMinificationFilter::LINEAR_MIPMAP_LINEAR,
                                       const TextureMagnificationFilter mag = TextureMagnificationFilter::LINEAR);
        std::shared_ptr<RawTextureData> _LoadTexture(const std::vector<std::string>&, 
                                                     const TextureHandle, 
                                                     const bool srgb,
                                                     const TextureType type = TextureType::TEXTURE_2D,
                                                     const TextureCoordinateWrapping wrap = TextureCoordinateWrapping::REPEAT,
                                                     const TextureMinificationFilter min = TextureMinificationFilter::LINEAR_MIPMAP_LINEAR,
                                                     const TextureMagnificationFilter mag = TextureMagnificationFilter::LINEAR);
        Texture * _FinalizeTexture(const RawTextureData&);

        void _InitCube();
        void _InitQuad();

    private:
        EntityPtr _cube;
        EntityPtr _quad;
        std::unordered_map<std::string, Async<Entity>> _loadedModels;
        std::unordered_map<std::string, Async<Entity>> _pendingFinalize;
        std::unordered_set<MeshPtr> _meshFinalizeQueue;
        std::unordered_map<TextureHandle, Async<RawTextureData>> _asyncLoadedTextureData;
        std::unordered_map<TextureHandle, Async<Texture>> _loadedTextures;
        std::unordered_map<std::string, TextureHandle> _loadedTexturesByFile;
        mutable std::shared_mutex _mutex;
    };
}