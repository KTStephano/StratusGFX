#include "StratusPipeline.h"
#include <StratusFilesystem.h>
#include <iostream>
#include "StratusLog.h"
#include <unordered_set>
#include "StratusUtils.h"
#include <sstream>
#include "StratusGraphicsDriver.h"

namespace stratus {
    bool ValidatePipeline(const Pipeline* p) {
        if (!p->IsValid()) {
            STRATUS_ERROR << p->GetError() << std::endl;
        }

        return p->IsValid();
    }

    bool ValidatePipeline(const Pipeline& p) {
        return ValidatePipeline(&p);
    }

    Pipeline::Pipeline(const std::filesystem::path& rootPath,
        const ShaderApiVersion& version,
        const std::vector<Shader>& shaders,
        const std::vector<std::pair<std::string, std::string>> defines)
        : shaders_(shaders), rootPath_(rootPath), version_(version), defines_(defines) {

        this->Compile_();
    }

    Pipeline::~Pipeline() {
        glDeleteProgram(program_);
    }

    static std::string BuildSourceWithLineNums(const std::string& source) {
        std::stringstream result;
        result << "==Begin Shader Source==" << std::endl;
        result << "1. ";
        usize lineNum = 2;
        for (usize i = 0; i < source.size(); ++i) {
            result << source[i];
            if (source[i] == '\n') {
                result << lineNum << ". ";
                ++lineNum;
            }
        }
        result << std::endl << "==End Shader Source==";

        return result.str();
    }

    static void PrintSourceWithLineNums(const std::string& source) {
        std::cout << BuildSourceWithLineNums(source) << std::endl;
    }

    /**
     * Takes a shader and checks if it raised any GL errors
     * during compilation. If so, it will print those errors
     * and return false.
     * @param shader shader to pull errors for
     * @return true if no errors occurred and false if anything
     *      went wrong
     */
    static bool checkShaderError(GLuint shader, const std::string& filename, const std::string& source, std::string& errorMessage) {
        errorMessage = "";
        GLint result;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &result);
        if (!result) {
            errorMessage += BuildSourceWithLineNums(source) + "\n";
            errorMessage += "[error] Unable to compile shader: " + filename + "\n";

            // Now we're going to get the error log and print it out
            GLint logLength;
            glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLength);
            if (logLength > 0) {
                std::string errorLog;
                errorLog.resize(static_cast<u32>(logLength));

                glGetShaderInfoLog(shader, logLength, nullptr, &errorLog[0]);
                errorMessage += errorLog + "\n";
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
    static bool checkProgramError(GLuint program, const std::vector<Shader>& shaders, std::string& errorMessage) {
        errorMessage = "";
        GLint linkStatus;
        glGetProgramiv(program, GL_LINK_STATUS, &linkStatus);
        if (!linkStatus) {
            std::string files;
            for (auto& s : shaders) files += s.filename + " ";
            errorMessage = "[error] Program failed during linking ( files were: " + files + ")" + "\n";

            GLint logLength;
            glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLength);

            if (logLength > 0) {
                std::string errorLog;
                errorLog.resize(static_cast<u32>(logLength));
                glGetProgramInfoLog(program, logLength, nullptr, &errorLog[0]);
                errorMessage += errorLog + "\n";
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

        for (usize i = 0; i < source.size(); ++i) {
            std::string line;
            bool foundWhitespace = false;
            // Read current line
            for (; i < source.size(); ++i) {
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
        for (usize i = 0; i < source.size(); ++i) {
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

    static void PreprocessShaderSource(std::string& source,
        const std::filesystem::path& root,
        const std::string& versionTag,
        const std::unordered_set<std::string>& allShaders,
        const std::vector<std::pair<std::string, std::string>>& defines) {
        PreprocessIncludes(source, root, allShaders);
        ReplaceFirst(source, "STRATUS_GLSL_VERSION", versionTag + "\n\nSTRATUS_GLSL_DEFINES");
        ReplaceAll(source, "STRATUS_GLSL_VERSION", "");
        // Build the define list
        std::string defineList;
        for (const auto& define : defines) {
            defineList = defineList + "#define " + define.first + " " + define.second + "\n";
        }
        ReplaceFirst(source, "STRATUS_GLSL_DEFINES", defineList);
    }

    void Pipeline::Compile_() {
        const std::unordered_set<std::string> allShaders = BuildFileList(rootPath_);
        const std::string versionTag = BuildShaderApiVersion(version_);

        isValid_ = true;
        std::vector<GLuint> shaderBinaries;
        for (Shader& s : this->shaders_) {
            const std::string shaderFile = rootPath_.string() + "/" + s.filename;
            STRATUS_LOG << "Loading shader: " << shaderFile << std::endl;
            std::string buffer = Filesystem::ReadAscii(shaderFile);
            if (buffer.empty()) {
                isValid_ = false;
                return;
            }

            PreprocessShaderSource(buffer, rootPath_, versionTag, allShaders, defines_);

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
                isValid_ = false;
                return;
            }

            GLuint bin = glCreateShader(type);
            const char* bufferPtr = buffer.c_str();
            glShaderSource(bin, 1, &bufferPtr, nullptr);
            glCompileShader(bin);

            if (!checkShaderError(bin, s.filename, buffer, error_)) {
                isValid_ = false;
                return;
            }

            shaderBinaries.push_back(bin);
        }

        // Link all the compiled binaries into a program
        program_ = glCreateProgram();
        for (auto bin : shaderBinaries) {
            glAttachShader(program_, bin);
        }
        glLinkProgram(program_);

        // We can safely delete the shaders now
        for (auto bin : shaderBinaries) {
            glDeleteShader(bin);
        }

        // Make sure no errors during linking came up
        if (!checkProgramError(program_, this->shaders_, error_)) {
            isValid_ = false;
            return;
        }
    }

    void Pipeline::Recompile() {
        Compile_();
    }

    void Pipeline::Bind() {
        Unbind();
        glUseProgram(program_);
    }

    void Pipeline::Unbind() {
        UnbindAllTextures();
        glUseProgram(0);
    }

    void Pipeline::SetBool(const std::string& uniform, bool b) const {
        SetInt(uniform, b ? 1 : 0);
    }

    void Pipeline::SetUint(const std::string& uniform, u32 i) const {
        glUniform1ui(GetUniformLocation(uniform), i);
    }

    void Pipeline::SetInt(const std::string& uniform, i32 i) const {
        glUniform1i(GetUniformLocation(uniform), i);
    }

    void Pipeline::SetFloat(const std::string& uniform, f32 f) const {
        glUniform1f(GetUniformLocation(uniform), f);
    }

    void Pipeline::SetUVec2(const std::string& uniform, const u32* vec, i32 num) const {
        glUniform2uiv(GetUniformLocation(uniform), num, vec);
    }

    void Pipeline::SetUVec3(const std::string& uniform, const u32* vec, i32 num) const {
        glUniform3uiv(GetUniformLocation(uniform), num, vec);
    }

    void Pipeline::SetIVec2(const std::string& uniform, const i32* vec, i32 num) const {
        glUniform2iv(GetUniformLocation(uniform), num, vec);
    }

    void Pipeline::SetIVec3(const std::string& uniform, const i32* vec, i32 num) const {
        glUniform3iv(GetUniformLocation(uniform), num, vec);
    }

    void Pipeline::SetIVec4(const std::string& uniform, const i32* vec, i32 num) const {
        glUniform4iv(GetUniformLocation(uniform), num, vec);
    }

    void Pipeline::SetUVec4(const std::string& uniform, const u32* vec, i32 num) const {
        glUniform4uiv(GetUniformLocation(uniform), num, vec);
    }

    void Pipeline::SetVec2(const std::string& uniform, const f32* vec, i32 num) const {
        glUniform2fv(GetUniformLocation(uniform), num, vec);
    }

    void Pipeline::SetVec3(const std::string& uniform, const f32* vec, i32 num) const {
        glUniform3fv(GetUniformLocation(uniform), num, vec);
    }

    void Pipeline::SetVec4(const std::string& uniform, const f32* vec, i32 num) const {
        glUniform4fv(GetUniformLocation(uniform), num, vec);
    }

    void Pipeline::SetMat2(const std::string& uniform, const f32* mat, i32 num) const {
        glUniformMatrix2fv(GetUniformLocation(uniform), num, GL_FALSE, mat);
    }

    void Pipeline::SetMat3(const std::string& uniform, const f32* mat, i32 num) const {
        glUniformMatrix3fv(GetUniformLocation(uniform), num, GL_FALSE, mat);
    }

    void Pipeline::SetMat4(const std::string& uniform, const f32* mat, i32 num) const {
        glUniformMatrix4fv(GetUniformLocation(uniform), num, GL_FALSE, mat);
    }

    void Pipeline::SetUVec2(const std::string& uniform, const glm::uvec2& v) const {
        SetUVec2(uniform, &v[0]);
    }

    void Pipeline::SetUVec3(const std::string& uniform, const glm::uvec3& v) const {
        SetUVec3(uniform, &v[0]);
    }

    void Pipeline::SetUVec4(const std::string& uniform, const glm::uvec4& v) const {
        SetUVec4(uniform, &v[0]);
    }

    void Pipeline::SetIVec2(const std::string& uniform, const glm::ivec2& v) const {
        SetIVec2(uniform, &v[0]);
    }

    void Pipeline::SetIVec3(const std::string& uniform, const glm::ivec3& v) const {
        SetIVec3(uniform, &v[0]);
    }

    void Pipeline::SetIVec4(const std::string& uniform, const glm::ivec4& v) const {
        SetIVec4(uniform, &v[0]);
    }

    void Pipeline::SetVec2(const std::string& uniform, const glm::vec2& v) const {
        SetVec2(uniform, (const f32*)&v[0]);
    }

    void Pipeline::SetVec3(const std::string& uniform, const glm::vec3& v) const {
        SetVec3(uniform, (const f32*)&v[0]);
    }

    void Pipeline::SetVec4(const std::string& uniform, const glm::vec4& v) const {
        SetVec4(uniform, (const f32*)&v[0]);
    }

    void Pipeline::SetMat2(const std::string& uniform, const glm::mat2& m) const {
        SetMat2(uniform, (const f32*)&m[0][0]);
    }

    void Pipeline::SetMat3(const std::string& uniform, const glm::mat3& m) const {
        SetMat3(uniform, (const f32*)&m[0][0]);
    }

    void Pipeline::SetMat4(const std::string& uniform, const glm::mat4& m) const {
        SetMat4(uniform, (const f32*)&m[0][0]);
    }

    GLint Pipeline::GetUniformLocation(const std::string& uniform) const {
        return glGetUniformLocation(program_, &uniform[0]);
    }

    GLint Pipeline::GetAttribLocation(const std::string& attrib) const {
        return glGetAttribLocation(program_, &attrib[0]);
    }

    std::vector<std::string> Pipeline::GetFileNames() const {
        std::vector<std::string> filenames;
        for (Shader s : this->shaders_) {
            filenames.push_back(s.filename);
        }
        return filenames;
    }

    void Pipeline::Print() const {
        auto& log = STRATUS_LOG;
        for (auto& s : GetFileNames()) {
            log << s << ", ";
        }
        log << std::endl;
    }

    void Pipeline::DispatchCompute(u32 xGroups, u32 yGroups, u32 zGroups) const {
        glDispatchCompute(xGroups, yGroups, zGroups);
    }

    void Pipeline::SynchronizeCompute() const {
        // See https://registry.khronos.org/OpenGL-Refpages/gl4/html/glBufferStorage.xhtml regarding GL_MAP_COHERENT_BIT
        // See https://registry.khronos.org/OpenGL-Refpages/gl4/html/glFenceSync.xhtml
        SynchronizeMemory();
        auto fence = HostInsertFence();
        HostFenceSync(fence);
    }

    void Pipeline::SynchronizeMemory() const {
        glMemoryBarrier(GL_ALL_BARRIER_BITS);
    }

    void Pipeline::BindTexture(const std::string& uniform, const Texture& tex) {
        const i32 activeTexture = NextTextureIndex_(uniform, tex);
        if (activeTexture < 0) return;

        tex.Bind(activeTexture);
        SetInt(uniform, activeTexture);
    }

    void Pipeline::BindTextureAsImage(const std::string& uniform, const Texture& tex, i32 mipLevel, bool layered, i32 layer, ImageTextureAccessMode access) {
        const i32 activeTexture = NextTextureIndex_(uniform, tex);
        if (activeTexture < 0) return;

        tex.BindAsImageTexture(activeTexture, mipLevel, layered, layer, access);
        SetInt(uniform, activeTexture);
    }

    void Pipeline::BindTextureAsImage(const std::string& uniform, const Texture& tex, i32 mipLevel, bool layered, i32 layer, ImageTextureAccessMode access, const TextureAccess& config) {
        const i32 activeTexture = NextTextureIndex_(uniform, tex);
        if (activeTexture < 0) return;

        tex.BindAsImageTexture(activeTexture, mipLevel, layered, layer, access, config);
        SetInt(uniform, activeTexture);
    }

    void Pipeline::UnbindAllTextures() {
        //for (auto & binding : boundTextures_) {
        //   binding.second.Unbind();
        //   SetInt(binding.first, 0);
        //}
        boundTextures_.clear();
        activeTextureIndices_.clear();
        activeTextureIndex_ = 0;
    }

    i32 Pipeline::NextTextureIndex_(const std::string& uniform, const Texture& tex) {
        if (!tex.Valid()) {
            STRATUS_ERROR << "[Error] Invalid texture passed to shader" << std::endl;
            return -1;
        }

        // See if the uniform is already bound to a texture
        auto it = activeTextureIndices_.find(uniform);
        if (it != activeTextureIndices_.end()) {
            //it->second.Unbind();
            //boundTextures_.find(uniform)->second.Unbind();
            boundTextures_.insert(std::make_pair(uniform, tex));
            return it->second;
        }

        auto next = activeTextureIndex_++;
        boundTextures_.insert(std::make_pair(uniform, tex));
        activeTextureIndices_.insert(std::make_pair(uniform, next));
        return next;
    }
}