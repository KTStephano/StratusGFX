#include "includes/Shader.h"
#include <includes/Filesystem.h>
#include <iostream>

Shader::Shader(const std::string &vertexShader,
        const std::string &fragShader)
        : _vsFile(vertexShader),
        _fsFile(fragShader) {
    _compile();
}

Shader::~Shader() {
    glDeleteProgram(_program);
}

/**
 * Takes a shader and checks if it raised any GL errors
 * during compilation. If so, it will print those errors
 * and return false.
 * @param shader shader to pull errors for
 * @return true if no errors occurred and false if anything
 *      went wrong
 */
static bool checkShaderError(GLuint shader) {
    GLint result;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &result);
    if (!result) {
        std::cerr << "[error] Unable to compile shader" << std::endl;

        // Now we're going to get the error log and print it out
        GLint logLength;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLength);
        if (logLength > 0) {
            std::string errorLog;
            errorLog.resize(static_cast<uint32_t>(logLength));

            glGetShaderInfoLog(shader, logLength, nullptr, &errorLog[0]);
            std::cout << errorLog << std::endl;
        }
        return false;
    }
    return true;
}

/**
 * Checks to see if any errors occured while linking
 * the shaders to the program.
 * @return true if nothing went wrong and false if anything
 *      bad happened
 */
static bool checkProgramError(GLuint program) {
    GLint linkStatus;
    glGetProgramiv(program, GL_LINK_STATUS, &linkStatus);
    if (!linkStatus) {
        std::cerr << "[error] Program failed during linking" << std::endl;

        GLint logLength;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLength);

        if (logLength > 0) {
            std::string errorLog;
            errorLog.resize(static_cast<uint32_t>(logLength));
            glGetProgramInfoLog(program, logLength, nullptr, &errorLog[0]);
            std::cout << errorLog << std::endl;
        }
        return false;
    }
    return true;
}

void Shader::_compile() {
    _isValid = true;
    std::string vsBuffer = Filesystem::readAscii(_vsFile);
    std::string fsBuffer = Filesystem::readAscii(_fsFile);
    if (vsBuffer.empty() || fsBuffer.empty()) {
        _isValid = false;
        return;
    }

    // Compile the vertex shader
    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    const char * bufferPtr = vsBuffer.c_str();
    glShaderSource(vs, 1, &bufferPtr, nullptr);
    glCompileShader(vs);

    if (!checkShaderError(vs)) {
        _isValid = false;
        return;
    }

    // Compile the fragment shader
    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    bufferPtr = fsBuffer.c_str();
    glShaderSource(fs, 1, &bufferPtr, nullptr);
    glCompileShader(fs);

    if (!checkShaderError(fs)) {
        // Make sure to delete vs since it succeeded
        glDeleteShader(vs);
        _isValid = false;
        return;
    }

    _program = glCreateProgram();
    glAttachShader(_program, vs);
    glAttachShader(_program, fs);
    // Compile program
    glLinkProgram(_program);

    // We can safely delete the shaders now
    glDeleteShader(vs);
    glDeleteShader(fs);
    if (!checkProgramError(_program)) {
        _isValid = false;
        return;
    }
}

void Shader::recompile() {
    _compile();
}

void Shader::bind() {
    glUseProgram(_program);
}

void Shader::unbind() {
    glUseProgram(0);
}

void Shader::setBool(const std::string &uniform, bool b) const {
    setInt(uniform, b ? 1 : 0);
}

void Shader::setInt(const std::string &uniform, int i) const {
    glUniform1i(getUniformLocation(uniform), i);
}

void Shader::setFloat(const std::string &uniform, float f) const {
    glUniform1f(getUniformLocation(uniform), f);
}

void Shader::setVec2(const std::string &uniform, const float *vec) const {
    glUniform2fv(getUniformLocation(uniform), 1, vec);
}

void Shader::setVec3(const std::string &uniform, const float *vec) const {
    glUniform3fv(getUniformLocation(uniform), 1, vec);
}

void Shader::setVec4(const std::string &uniform, const float *vec) const {
    glUniform4fv(getUniformLocation(uniform), 1, vec);
}

void Shader::setMat2(const std::string &uniform, const float *mat) const {
    glUniformMatrix2fv(getUniformLocation(uniform), 1, GL_FALSE, mat);
}

void Shader::setMat3(const std::string &uniform, const float *mat) const {
    glUniformMatrix3fv(getUniformLocation(uniform), 1, GL_FALSE, mat);
}

void Shader::setMat4(const std::string &uniform, const float *mat) const {
    glUniformMatrix4fv(getUniformLocation(uniform), 1, GL_FALSE, mat);
}

GLint Shader::getUniformLocation(const std::string &uniform) const {
    return glGetUniformLocation(_program, &uniform[0]);
}

GLint Shader::getAttribLocation(const std::string &attrib) const {
    return glGetAttribLocation(_program, &attrib[0]);
}