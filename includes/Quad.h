

#ifndef STRATUSGFX_QUAD_H
#define STRATUSGFX_QUAD_H

#include "RenderEntity.h"

class Quad : public RenderEntity {
    GLuint _vao;
    GLuint _buffer;

public:
    Quad(RenderMode);
    virtual ~Quad();
    void render() override;
};

#endif //STRATUSGFX_QUAD_H
