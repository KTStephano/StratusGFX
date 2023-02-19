#include "StratusPipeline.h"
#include <StratusFilesystem.h>
#include <iostream>
#include "StratusLog.h"
#include <unordered_set>
#include "StratusUtils.h"

namespace stratus {
Pipeline::Pipeline(const std::filesystem::path& rootPath, const ShaderApiVersion& version, const std::vector<Shader> & shaders)
    : _shaders(shaders), _rootPath(rootPath), _version(version) {

    this->_compile();
}

Pipeline::~Pipeline() {
    glDeleteProgram(_program);
}

static void PrintSourceWithLineNums(const std::string& source) {
    std::cout << "==Begin Shader Source==" << std::endl;
    std::cout << "1. ";
    size_t lineNum = 2;
    for (size_t i = 0; i < source.size(); ++i) {
        std::cout << source[i];
        if (source[i] == '\n') {
            std::cout << lineNum << ". ";
            ++lineNum;
        }
    }
    std::cout << std::endl << "==End Shader Source==" << std::endl;
}

/**
 * Takes a shader and checks if it raised any GL errors
 * during compilation. If so, it will print those errors
 * and return false.
 * @param shader shader to pull errors for
 * @return true if no errors occurred and false if anything
 *      went wrong
 */
static bool checkShaderError(GLuint shader, const std::string & filename, const std::string & source) {
    GLint result;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &result);
    if (!result) {
        PrintSourceWithLineNums(source);

        STRATUS_ERROR << "[error] Unable to compile shader: " << filename << std::endl;

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
static bool checkProgramError(GLuint program, const std::vector<Shader> & shaders) {
    GLint linkStatus;
    glGetProgramiv(program, GL_LINK_STATUS, &linkStatus);
    if (!linkStatus) {
        std::string files;
        for (auto& s : shaders) files += s.filename + " ";
        STRATUS_ERROR << "[error] Program failed during linking ( files were: " << files << ")" << std::endl;

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

static std::unordered_set<std::string> BuildFileList(const std::filesystem::path& root) {
    std::unordered_set<std::string> result;
    std::string rootReformatted = root.string();
    ReplaceAll(rootReformatted, "\\", "/");

    for (const std::filesystem::directory_entry& entry : std::filesystem::recursive_directory_iterator(root)) {
        if (entry.is_regular_file()) {
            std::string file = entry.path().string();
            ReplaceAll(file, "\\", "/");
            ReplaceFirst(file, rootReformatted + "/", "");
            result.insert(file);
        }
    }

    return result;
}

static std::string BuildShaderApiVersion(const ShaderApiVersion& version) {
    std::string result = "#version ";

    if (version.major <= 3 && version.minor < 3) {
        result += "GL_VERSION_UNSUPPORTED_TOO_OLD";
        return result;
    }

    result += std::to_string(version.major) + std::to_string(version.minor) + "0 core";
    return result;
}

static std::string ExtractFirstInclude(const std::string& source) {
    std::string result;

    for (size_t i = 0; i < source.size(); ++i) {
        std::string line;
        bool foundWhitespace = false;
        // Read current line
        for( ; i < source.size(); ++i) {
            const char c = source[i];
            if (c == '\n') {
                break;
            }

            const bool isWhitespace = c == ' ' || c == '\t' || c == '\r';
            // We skip all the initial whitespace until we hit the first valid character
            if (!isWhitespace) {
                // Current character is valid - stop looking for whitespace
                foundWhitespace = true;
            }
            else if (foundWhitespace == false) {
                foundWhitespace = true;
                continue;
            }

            line += c;
        }

        if (BeginsWith(line, "#include")) {
            result = line;
            break;
        }
    }

    return result;
}

static std::string ExtractIncludeFile(const std::string& source) {
    std::string file;
    bool foundQuote = false;
    for (size_t i = 0; i < source.size(); ++i) {
        const char c = source[i];
        if (c == '\"') {
            if (foundQuote) {
                break;
            }
            foundQuote = true;
        }
        else if (foundQuote) {
            file += c;
        }
    }

    return file;
}

static void PreprocessIncludes(std::string& source, const std::filesystem::path& root, const std::unordered_set<std::string>& allShaders) {
    std::unordered_set<std::string> seenIncludes;
    // We need to do this in a loop since bringing in one file could also bring in additional includes
    while (true) {
        std::string line = ExtractFirstInclude(source);
        // Reached the end of the include list
        if (line.empty()) {
            break;
        }

        const std::string file = ExtractIncludeFile(line);
        if (seenIncludes.find(file) == seenIncludes.end()) {
            seenIncludes.insert(file);
            const std::string fullPath = root.string() + "/" + file;
            std::string includeSource = Filesystem::ReadAscii(fullPath);
            ReplaceFirst(source, line, includeSource);
        }
        else {
            ReplaceAll(source, line, "");
        }
    }
}

static void PreprocessShaderSource(std::string& source, const std::filesystem::path& root, const std::string& versionTag, const std::unordered_set<std::string>& allShaders) {
    PreprocessIncludes(source, root, allShaders);
    ReplaceFirst(source, "STRATUS_GLSL_VERSION", versionTag);
    ReplaceAll(source, "STRATUS_GLSL_VERSION", "");
}

void Pipeline::_compile() {
    const std::unordered_set<std::string> allShaders = BuildFileList(_rootPath);
    const std::string versionTag = BuildShaderApiVersion(_version);

    _isValid = true;
    std::vector<GLuint> shaderBinaries;
    for (Shader & s : this->_shaders) {
        const std::string shaderFile = _rootPath.string() + "/" + s.filename;
        STRATUS_LOG << "Loading shader: " << shaderFile << std::endl;
        std::string buffer = Filesystem::ReadAscii(shaderFile);
        if (buffer.empty()) {
            _isValid = false;
            return;
        }

        PreprocessShaderSource(buffer, _rootPath, versionTag, allShaders);

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
            STRATUS_ERROR << "Unknown shader type" << std::endl;
            _isValid = false;
            return;
        }

        GLuint bin = glCreateShader(type);
        const char * bufferPtr = buffer.c_str();
        glShaderSource(bin, 1, &bufferPtr, nullptr);
        glCompileShader(bin);

        if (!checkShaderError(bin, s.filename, buffer)) {
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
    if (!checkProgramError(_program, this->_shaders)) {
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

void Pipeline::setVec2(const std::string & uniform, const glm::vec2& v) const {
    setVec2(uniform, (const float *)&v[0]);
}

void Pipeline::setVec3(const std::string & uniform, const glm::vec3& v) const {
    setVec3(uniform, (const float *)&v[0]);
}

void Pipeline::setVec4(const std::string & uniform, const glm::vec4& v) const {
    setVec4(uniform, (const float *)&v[0]);
}

void Pipeline::setMat2(const std::string & uniform, const glm::mat2& m) const {
    setMat2(uniform, (const float *)&m[0][0]);
}

void Pipeline::setMat3(const std::string & uniform, const glm::mat3& m) const {
    setMat3(uniform, (const float *)&m[0][0]);
}

void Pipeline::setMat4(const std::string & uniform, const glm::mat4& m) const {
    setMat4(uniform, (const float *)&m[0][0]);
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
    auto & log = STRATUS_LOG;
    for (auto & s : getFileNames()) {
        log << s << ", ";
    }
    log << std::endl;
}

void Pipeline::dispatchCompute(unsigned int xGroups, unsigned int yGroups, unsigned int zGroups) {
    glDispatchCompute(xGroups, yGroups, zGroups);
}

void Pipeline::synchronizeCompute() {
    // See https://registry.khronos.org/OpenGL-Refpages/gl4/html/glBufferStorage.xhtml regarding GL_MAP_COHERENT_BIT
    // See https://registry.khronos.org/OpenGL-Refpages/gl4/html/glFenceSync.xhtml
    glMemoryBarrier(GL_ALL_BARRIER_BITS);
    glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
}

void Pipeline::bindTexture(const std::string & uniform, const Texture & tex) {
    if (!tex.valid()) {
        STRATUS_ERROR << "[Error] Invalid texture passed to shader" << std::endl;
        return;
    }
    // See if the uniform is already bound to a texture
    auto it = _boundTextures.find(uniform);
    if (it != _boundTextures.end()) {
        it->second.unbind();
    }

    const int activeTexture = _activeTextureIndex++;
    tex.bind(activeTexture);
    setInt(uniform, activeTexture);
    _boundTextures.insert(std::make_pair(uniform, tex));
}

void Pipeline::unbindAllTextures() {
    for (auto & binding : _boundTextures) {
        binding.second.unbind();
        setInt(binding.first, 0);
    }
    _boundTextures.clear();
    _activeTextureIndex = 1;
}
}