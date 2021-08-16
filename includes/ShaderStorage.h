#pragma once

#include "GL/gl3w.h"
#include <vector>

namespace stratus {
    /**
     * Shader storage is a type of general-purpose memory that lives in the GPU
     * memory pool. When needed it can be mapped and unmapped for read/write operations.
     */
    template<typename E>
    class ShaderStorage {
        // Handle to GPU storage
        GLuint _buffer = 0;

        // Number of elements (to get bytes, sizeof(E) * size)
        size_t _size = 0;

        // Only valid after a call to mmap()
        E * _ptr = nullptr;

        // Cached from a call to bind()
        int _index = -1;

    public:
        ShaderStorage(const size_t size) {
            glGenBuffers(1, &_buffer);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, _buffer);
            glBufferData(GL_SHADER_STORAGE_BUFFER,  size * sizeof(E), nullptr, GL_STATIC_DRAW); 
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
            _size = size;
        }

        ShaderStorage(const std::vector<E> & data) {
            glGenBuffers(1, &_buffer);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, _buffer);
            glBufferData(GL_SHADER_STORAGE_BUFFER, data.size() * sizeof(E), &data[0], GL_STATIC_DRAW);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
            _size = data.size();
        }

        ShaderStorage(const ShaderStorage &) = delete;
        ShaderStorage(ShaderStorage &&) = default;

        ~ShaderStorage() {
            _destroy();
        }

        ShaderStorage & operator=(const ShaderStorage &) = delete;
        ShaderStorage & operator=(ShaderStorage && other) {
            _destroy();

            _buffer = other._buffer;
            _size = other._size;
            _ptr = other._ptr;
            _index = other._index;

            other._buffer = 0;
            other._size = 0;
            other._ptr = nullptr;
            other._index = -1;

            return *this;
        }

        void mmap() {
            if (_ptr != nullptr) return;
            glBindBuffer(GL_ARRAY_BUFFER, _buffer);
            _ptr = reinterpret_cast<E *>(glMapBufferRange(GL_ARRAY_BUFFER, 0, _size * sizeof(E), GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT));
        }

        void munmap() {
            if (_ptr == nullptr) return;
            glUnmapBuffer(GL_ARRAY_BUFFER);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
        }

        size_t size() const {
            return _size;
        }

        void bind(int index) {
            if (index != -1) unbind();
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, index, _buffer);
            _index = index;
        }

        void unbind() {
            if (index == -1) return;
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, _index, 0);
            _index = -1;
        }

        const E * ptr() const {
            return _ptr;
        }

        E * ptr() {
            return _ptr;
        }

    private:
        void _destroy() {
            if (this->_buffer == 0) return;
            munmap();
            glDeleteBuffers(1, &_buffer);
            _buffer = 0;
            _size = 0;
        }
    };
}