
#include <includes/RenderEntity.h>

#include "includes/RenderEntity.h"
#include "includes/Common.h"

RenderEntity::RenderEntity(RenderMode mode,
        RenderProperties properties)
        : _mode(mode) {
    setRenderProperties(properties);
}

RenderEntity::~RenderEntity() {

}

void RenderEntity::setRenderProperties(RenderProperties properties) {
    _properties = properties;
}

void RenderEntity::appentRenderProperties(RenderProperties properties) {
    auto p = uint32_t(properties);
    auto current = uint32_t(_properties);
    _properties = (RenderProperties)(current | p);
}

RenderMode RenderEntity::getRenderMode() const {
    return _mode;
}

RenderProperties RenderEntity::getRenderProperties() const {
    return _properties;
}
