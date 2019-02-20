

#ifndef STRATUSGFX_QUAD_H
#define STRATUSGFX_QUAD_H

#include "RenderEntity.h"

class Quad : public RenderEntity {
    GLuint _vao;
    GLuint _buffer;

public:
    Quad();
    virtual ~Quad();
    void render() override;
};

#endif //STRATUSGFX_QUAD_H
