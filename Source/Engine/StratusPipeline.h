
#ifndef STRATUSGFX_Pipeline_H
#define STRATUSGFX_Pipeline_H

#include <string>
#include "GL/gl3w.h"
#include <vector>
#include "StratusTexture.h"
#include <unordered_map>
#include "glm/glm.hpp"
#include <filesystem>

namespace stratus {
    enum class ShaderType {
        VERTEX,
        GEOMETRY,
        FRAGMENT,
        COMPUTE
    };

    struct Shader {
        std::string filename;
        ShaderType type;
    };

    struct ShaderApiVersion {
        int major;
        int minor;
    };

class Pipeline {
    /**
     * List of all shaders used by the pipeline.
     */
    std::vector<Shader> shaders_;
    
    /**
     * Specifies the top level directory where all shaders are located.
     */
    std::filesystem::path rootPath_;

    /**
     * Contains information about the graphics API version.
     */
    ShaderApiVersion version_;

    // List of #defines for the shader
    std::vector<std::pair<std::string, std::string>> defines_;

    // List of bound textures since the last call to bind()
    std::unordered_map<std::string, Texture> boundTextures_;

    // Lets us keep track of the next texture index to use
    int activeTextureIndex_ = 0;

    /**
     * Program handle returned from OpenGL
     */
    GLuint program_;

    /**
     * Used to determine whether or not this Pipeline
     * is valid. If true then it is safe to use.
     */
    bool isValid_ = false;

public:
    /**
     * @param vertexPipeline file for the vertex Pipeline
     * @param geomPipeline file for the geometry Pipeline (optional)
     * @param fragPipeline file for the fragment Pipeline
     */
    Pipeline(const std::filesystem::path& rootPath, 
             const ShaderApiVersion& version, 
             const std::vector<Shader>& shaders,
             const std::vector<std::pair<std::string, std::string>> defines = {});
    ~Pipeline();

    /**
     * @return true if the Pipeline was successfully compiled
     */
    bool IsValid() { return isValid_; }

    /**
     * Tells the Pipeline to recompile its source files.
     */
    void Recompile();

    /**
     * Binds this Pipeline so that it can be used for rendering.
     */
    void Bind();

    /**
     * Unbinds this Pipeline so that it no longer affects future
     * rendering.
     */
    void Unbind();

    /**
     * Takes a uniform name (such as "viewMatrix") and returns its
     * location within the Pipeline.
     * @param uniform name of the uniform
     * @return integer representing the uniform location
     */
    GLint GetUniformLocation(const std::string & uniform) const;
    GLint GetAttribLocation(const std::string & attrib) const;

    std::vector<std::string> GetFileNames() const;
    void Print() const;

    /**
     * Functions for use with compute programs.
     */
    // Here x/y/zGroups specify work group units, so if they are defined by (local_size_x = 32)
    // then passing 2 for xGroups would result in 2 * 32 = 64 invokations
    void DispatchCompute(unsigned int xGroups, unsigned int yGroups, unsigned int zGroups);
    void SynchronizeCompute();

    /**
     * Various setters to make it easy to set various uniforms
     * such as bool, int, float, vector, matrix.
     */
     void SetBool(const std::string & uniform, bool b) const;
     void SetUint(const std::string& uniform, unsigned int i) const;
     void SetInt(const std::string & uniform, int i) const;
     void SetFloat(const std::string & uniform, float f) const;
     void SetUVec2(const std::string & uniform, const unsigned int * vec, int num = 1) const;
     void SetUVec3(const std::string & uniform, const unsigned int * vec, int num = 1) const;
     void SetUVec4(const std::string & uniform, const unsigned int * vec, int num = 1) const;
     void SetVec2(const std::string & uniform, const float * vec, int num = 1) const;
     void SetVec3(const std::string & uniform, const float * vec, int num = 1) const;
     void SetVec4(const std::string & uniform, const float * vec, int num = 1) const;
     void SetMat2(const std::string & uniform, const float * mat, int num = 1) const;
     void SetMat3(const std::string & uniform, const float * mat, int num = 1) const;
     void SetMat4(const std::string & uniform, const float * mat, int num = 1) const;

     void SetUVec2(const std::string & uniform, const glm::uvec2&) const;
     void SetUVec3(const std::string & uniform, const glm::uvec3&) const;
     void SetUVec4(const std::string & uniform, const glm::uvec4&) const;
     void SetVec2(const std::string & uniform, const glm::vec2&) const;
     void SetVec3(const std::string & uniform, const glm::vec3&) const;
     void SetVec4(const std::string & uniform, const glm::vec4&) const;
     void SetMat2(const std::string & uniform, const glm::mat2&) const;
     void SetMat3(const std::string & uniform, const glm::mat3&) const;
     void SetMat4(const std::string & uniform, const glm::mat4&) const;

     // Texture management
     void BindTexture(const std::string & uniform, const Texture & tex);
     void UnbindAllTextures();

private:
    void Compile_();
};
}

#endif //STRATUSGFX_Pipeline_H