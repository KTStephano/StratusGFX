#include "Pipeline.h"
#include <Filesystem.h>
#include <iostream>

namespace stratus {
Pipeline::Pipeline(const std::vector<Shader> & shaders)
    : _shaders(shaders) {
    this->_compile();
}

Pipeline::~Pipeline() {
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
static bool checkShaderError(GLuint shader, const std::string & filename) {
    GLint result;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &result);
    if (!result) {
        std::cerr << "[error] Unable to compile shader: " << filename << std::endl;

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

void Pipeline::_compile() {
    _isValid = true;
    std::vector<GLuint> shaderBinaries;
    for (Shader & s : this->_shaders) {
        std::string buffer = Filesystem::readAscii(s.filename);
        if (buffer.empty()) {
            _isValid = false;
            return;
        }

        GLenum type;
        switch (s.type) {
        case ShaderType::VERTEX:
            type = GL_VERTEX_SHADER;
            break;
        case ShaderType::GEOMETRY:
            type = GL_GEOMETRY_SHADER;
            break;
        case ShaderType::FRAGMENT:
            type = GL_FRAGMENT_SHADER;
            break;
        case ShaderType::COMPUTE:
            type = GL_COMPUTE_SHADER;
            break;
        default:
            std::cerr << "Unknown shader type" << std::endl;
            _isValid = false;
            return;
        }

        GLuint bin = glCreateShader(type);
        const char * bufferPtr = buffer.c_str();
        glShaderSource(bin, 1, &bufferPtr, nullptr);
        glCompileShader(bin);

        if (!checkShaderError(bin, s.filename)) {
            _isValid = false;
            return;
        }

        shaderBinaries.push_back(bin);
    }

    // Link all the compiled binaries into a program
    _program = glCreateProgram();
    for (auto bin : shaderBinaries) {
        glAttachShader(_program, bin);
    }
    glLinkProgram(_program);

    // We can safely delete the shaders now
    for (auto bin : shaderBinaries) {
        glDeleteShader(bin);
    }

    // Make sure no errors during linking came up
    if (!checkProgramError(_program)) {
        _isValid = false;
        return;
    }
}

void Pipeline::recompile() {
    _compile();
}

void Pipeline::bind() {
    glUseProgram(_program);
}

void Pipeline::unbind() {
    unbindAllTextures();
    glUseProgram(0);
}

void Pipeline::setBool(const std::string &uniform, bool b) const {
    setInt(uniform, b ? 1 : 0);
}

void Pipeline::setInt(const std::string &uniform, int i) const {
    glUniform1i(getUniformLocation(uniform), i);
}

void Pipeline::setFloat(const std::string &uniform, float f) const {
    glUniform1f(getUniformLocation(uniform), f);
}

void Pipeline::setVec2(const std::string &uniform, const float *vec, int num) const {
    glUniform2fv(getUniformLocation(uniform), num, vec);
}

void Pipeline::setVec3(const std::string &uniform, const float *vec, int num) const {
    glUniform3fv(getUniformLocation(uniform), num, vec);
}

void Pipeline::setVec4(const std::string &uniform, const float *vec, int num) const {
    glUniform4fv(getUniformLocation(uniform), num, vec);
}

void Pipeline::setMat2(const std::string &uniform, const float *mat, int num) const {
    glUniformMatrix2fv(getUniformLocation(uniform), num, GL_FALSE, mat);
}

void Pipeline::setMat3(const std::string &uniform, const float *mat, int num) const {
    glUniformMatrix3fv(getUniformLocation(uniform), num, GL_FALSE, mat);
}

void Pipeline::setMat4(const std::string &uniform, const float *mat, int num) const {
    glUniformMatrix4fv(getUniformLocation(uniform), num, GL_FALSE, mat);
}

GLint Pipeline::getUniformLocation(const std::string &uniform) const {
    return glGetUniformLocation(_program, &uniform[0]);
}

GLint Pipeline::getAttribLocation(const std::string &attrib) const {
    return glGetAttribLocation(_program, &attrib[0]);
}

std::vector<std::string> Pipeline::getFileNames() const {
    std::vector<std::string> filenames;
    for (Shader s : this->_shaders) {
        filenames.push_back(s.filename);
    }
    return filenames;
}

void Pipeline::print() const {
    for (auto & s : getFileNames()) {
        std::cout << s << ", ";
    }
    std::cout << std::endl;
}

void Pipeline::bindTexture(const std::string & uniform, const Texture & tex) {
    if (!tex.valid()) {
        std::cerr << "[Error] Invalid texture passed to shader" << std::endl;
        return;
    }
    const int activeTexture = _boundTextures.size();
    setInt(uniform, activeTexture);
    tex.bind(activeTexture);
    _boundTextures.push_back(TextureBinding{uniform, tex});
}

void Pipeline::unbindAllTextures() {
    for (const TextureBinding & binding : _boundTextures) {
        binding.texture.unbind();
        setInt(binding.uniform, 0);
    }
    _boundTextures.clear();
}
}