
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
    std::vector<Shader> _shaders;

    // List of bound textures since the last call to bind()
    std::unordered_map<std::string, Texture> _boundTextures;

    // Lets us keep track of the next texture index to use
    int _activeTextureIndex = 1;

    /**
     * Program handle returned from OpenGL
     */
    GLuint _program;

    /**
     * Used to determine whether or not this Pipeline
     * is valid. If true then it is safe to use.
     */
    bool _isValid = false;

    /**
     * Specifies the top level directory where all shaders are located.
     */
    std::filesystem::path _rootPath;

    /**
     * Contains information about the graphics API version.
     */
    ShaderApiVersion _version;

public:
    /**
     * @param vertexPipeline file for the vertex Pipeline
     * @param geomPipeline file for the geometry Pipeline (optional)
     * @param fragPipeline file for the fragment Pipeline
     */
    Pipeline(const std::filesystem::path& rootPath, const ShaderApiVersion& version, const std::vector<Shader>& shaders);
    ~Pipeline();

    /**
     * @return true if the Pipeline was successfully compiled
     */
    bool isValid() { return _isValid; }

    /**
     * Tells the Pipeline to recompile its source files.
     */
    void recompile();

    /**
     * Binds this Pipeline so that it can be used for rendering.
     */
    void bind();

    /**
     * Unbinds this Pipeline so that it no longer affects future
     * rendering.
     */
    void unbind();

    /**
     * Takes a uniform name (such as "viewMatrix") and returns its
     * location within the Pipeline.
     * @param uniform name of the uniform
     * @return integer representing the uniform location
     */
    GLint getUniformLocation(const std::string & uniform) const;
    GLint getAttribLocation(const std::string & attrib) const;

    std::vector<std::string> getFileNames() const;
    void print() const;

    /**
     * Various setters to make it easy to set various uniforms
     * such as bool, int, float, vector, matrix.
     */
     void setBool(const std::string & uniform, bool b) const;
     void setInt(const std::string & uniform, int i) const;
     void setFloat(const std::string & uniform, float f) const;
     void setVec2(const std::string & uniform, const float * vec, int num = 1) const;
     void setVec3(const std::string & uniform, const float * vec, int num = 1) const;
     void setVec4(const std::string & uniform, const float * vec, int num = 1) const;
     void setMat2(const std::string & uniform, const float * mat, int num = 1) const;
     void setMat3(const std::string & uniform, const float * mat, int num = 1) const;
     void setMat4(const std::string & uniform, const float * mat, int num = 1) const;

     void setVec2(const std::string & uniform, const glm::vec2&) const;
     void setVec3(const std::string & uniform, const glm::vec3&) const;
     void setVec4(const std::string & uniform, const glm::vec4&) const;
     void setMat2(const std::string & uniform, const glm::mat2&) const;
     void setMat3(const std::string & uniform, const glm::mat3&) const;
     void setMat4(const std::string & uniform, const glm::mat4&) const;

     // Texture management
     void bindTexture(const std::string & uniform, const Texture & tex);
     void unbindAllTextures();

private:
    void _compile();
};
}

#endif //STRATUSGFX_Pipeline_H