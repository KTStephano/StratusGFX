
#ifndef STRATUSGFX_SHADER_H
#define STRATUSGFX_SHADER_H

#include <string>
#include "GL/gl.h"

class Shader {
    /**
     * Filename for the vertex shader
     */
    std::string _vtFile;

    /**
     * Filename for the fragment shader
     */
    std::string _fsFile;

    /**
     * Shader handle returned from OpenGL
     */
    GLuint _shader;

    /**
     * Used to determine whether or not this shader
     * is valid. If true then it is safe to use.
     */
    bool _isValid = false;

public:
    /**
     * @param vertexShader file for the vertex shader
     * @param fragShader file for the fragment shader
     */
    Shader(const std::string & vertexShader,
            const std::string & fragShader);
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

private:
    bool _compile();
};

#endif //STRATUSGFX_SHADER_H
