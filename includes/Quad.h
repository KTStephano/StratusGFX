

#ifndef STRATUSGFX_QUAD_H
#define STRATUSGFX_QUAD_H

#include "RenderEntity.h"

class Quad : public RenderEntity {
    GLuint _vao;
    GLuint _buffer;

public:
    Quad();
    Quad(const Quad & other) = delete;
    Quad(Quad && other) = delete;
    Quad & operator=(const Quad & other) = delete;
    Quad & operator=(Quad && other) = delete;
    ~Quad() override;
    void render() override;
    void renderInstanced(const int) override;
};

#endif //STRATUSGFX_QUAD_H
