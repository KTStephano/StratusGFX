
#include <RenderEntity.h>

#include "RenderEntity.h"
#include "Common.h"

namespace stratus {
RenderEntity::RenderEntity(RenderProperties properties) {
    _setProperties(properties);
}

RenderEntity::~RenderEntity() {

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
    if (material.normalMap == -1) {
        _disableProperties(NORMAL_MAPPED);
        _disableProperties(NORMAL_HEIGHT_MAPPED);
    }
    else {
        if (material.depthMap == -1) {
            _disableProperties(NORMAL_HEIGHT_MAPPED);
            _enableProperties(NORMAL_MAPPED);
        }
        else {
            _disableProperties(NORMAL_MAPPED);
            _enableProperties(NORMAL_HEIGHT_MAPPED);
        }
    }
}

const RenderMaterial &RenderEntity::getMaterial() const {
    return _material;
}

const RenderProperties &RenderEntity::getRenderProperties() const {
    return _properties;
}

const RenderData &RenderEntity::getRenderData() const {
    return _data;
}
}