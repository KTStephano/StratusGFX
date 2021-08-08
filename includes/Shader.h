
#ifndef STRATUSGFX_SHADER_H
#define STRATUSGFX_SHADER_H

#include <string>
#include "GL/gl3w.h"

namespace stratus {
class Shader {
    /**
     * Filename for the vertex shader
     */
    std::string _vsFile;

    /**
     * Filename for the geometry shader (optional - may be empty)
     */
    std::string _gsFile;

    /**
     * Filename for the fragment shader
     */
    std::string _fsFile;

    /**
     * Program handle returned from OpenGL
     */
    GLuint _program;

    /**
     * Used to determine whether or not this shader
     * is valid. If true then it is safe to use.
     */
    bool _isValid = false;

public:
    /**
     * @param vertexShader file for the vertex shader
     * @param geomShader file for the geometry shader (optional)
     * @param fragShader file for the fragment shader
     */
    Shader(const std::string & vertexShader, const std::string & geomShader, const std::string & fragShader);
    Shader(const std::string & vertexShader, const std::string & fragShader);
    ~Shader();

    /**
     * @return true if the shader was successfully compiled
     */
    bool isValid() { return _isValid; }

    /**
     * Tells the shader to recompile its source files.
     */
    void recompile();

    /**
     * Binds this shader so that it can be used for rendering.
     */
    void bind();

    /**
     * Unbinds this shader so that it no longer affects future
     * rendering.
     */
    void unbind();

    /**
     * Takes a uniform name (such as "viewMatrix") and returns its
     * location within the shader.
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

private:
    void _compile();
};
}

#endif //STRATUSGFX_SHADER_H