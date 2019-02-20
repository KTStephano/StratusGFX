
#include <includes/RenderEntity.h>

#include "includes/RenderEntity.h"
#include "includes/Common.h"

RenderEntity::RenderEntity(RenderProperties properties) {
    _setProperties(properties);
}

RenderEntity::~RenderEntity() {

}

RenderProperties RenderEntity::getRenderProperties() const {
    return _properties;
}

void RenderEntity::_setProperties(uint32_t properties) {
    _properties = (RenderProperties)properties;
}

void RenderEntity::_enableProperties(uint32_t properties) {
    auto p = (uint32_t)_properties;
    p = p | properties;
    _properties = (RenderProperties)p;
}

void RenderEntity::_disableProperties(uint32_t properties) {
    auto p = (uint32_t)_properties;
    p = p & (~properties);
    _properties = (RenderProperties)p;
}

void RenderEntity::enableInvisibility(bool invisible) {
    if (invisible) _disableProperties(INVISIBLE);
    else _enableProperties(INVISIBLE);
}

void RenderEntity::enableLightInteraction(bool enabled) {
    if (enabled) {
        _disableProperties(FLAT);
        _enableProperties(DYNAMIC);
    } else {
        _disableProperties(DYNAMIC);
        _enableProperties(FLAT);
    }
}

void RenderEntity::setMaterial(const RenderMaterial &material) {
    _material = material;
    if (material.texture == -1) {
        _disableProperties(TEXTURED);
    } else {
        _enableProperties(TEXTURED);
    }
}

const RenderMaterial &RenderEntity::getMaterial() const {
    return _material;
}
