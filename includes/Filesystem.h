//
// Created by stephano on 2/17/19.
//

#ifndef STRATUSGFX_FILESYSTEM_H
#define STRATUSGFX_FILESYSTEM_H

#include <vector>
#include <string>

class Filesystem {
    /**
     * Reads a binary file and returns an array of bytes.
     */
    static std::vector<uint8_t> readBinary(const std::string & file);

    /**
     * Reads a file interpreted as ASCII and returns a string with
     * all of its contents.
     */
    static std::string readAscii(const std::string & file);
};

#endif //STRATUSGFX_FILESYSTEM_H
