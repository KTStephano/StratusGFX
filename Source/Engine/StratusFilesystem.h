

#ifndef STRATUSGFX_FILESYSTEM_H
#define STRATUSGFX_FILESYSTEM_H

#include <vector>
#include <string>
#include <filesystem>

namespace stratus {
    struct Filesystem {
        /**
         * Reads a binary file and returns an array of bytes.
         */
        static std::vector<char> ReadBinary(const std::string & file);

        /**
         * Reads a file interpreted as ASCII and returns a string with
         * all of its contents.
         */
        static std::string ReadAscii(const std::string & file);

        // Returns the current working directory
        static std::filesystem::path CurrentPath();
    };
}

#endif //STRATUSGFX_FILESYSTEM_H
