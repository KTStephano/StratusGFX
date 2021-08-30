
#include <StratusCamera.h>
#include <iostream>

#include "StratusCamera.h"
#include "StratusUtils.h"

namespace stratus {
    Camera::Camera(bool rangeCheckAngles) : _rangeCheckAngles(rangeCheckAngles) {}

    void Camera::modifyAngle(Degrees deltaYaw, Degrees deltaPitch, Degrees deltaRoll) {
        setAngle(Rotation(_rotation.x + deltaYaw, _rotation.y + deltaPitch, _rotation.z + deltaRoll));
    }

    void Camera::setAngle(const Rotation & rotation) {
        _rotation = rotation;
        if (_rangeCheckAngles) {
            if (_rotation.x.value() > 89) _rotation.x = Degrees(89.0f);
            else if (_rotation.x.value() < -89.0f) _rotation.x = Degrees(-89.0f);
        }
        _invalidateView();
    }

    const Rotation & Camera::getRotation() const {
        return _rotation;
    }

    void Camera::setPosition(float x, float y, float z) {
        setPosition(glm::vec3(x, y, z));
    }

    void Camera::setPosition(const glm::vec3 & position) {
        _position = position;
        _invalidateView();
    }

    const glm::vec3 & Camera::getPosition() const {
        return _position;
    }

    glm::vec3 Camera::getDirection() const {
        return glm::normalize(-getWorldTransform()[2]);
    }

    glm::vec3 Camera::getUp() const {
        return glm::normalize(-getWorldTransform()[1]);
    }

    glm::vec3 Camera::getSide() const {
        return -glm::cross(getDirection(), getUp());
    }

    void Camera::setSpeed(float forward, float up, float strafe) {
        setSpeed(glm::vec3(forward, up, strafe));
    }

    void Camera::setSpeed(const glm::vec3 & speed) {
        _speed = speed;
    }

    const glm::vec3 & Camera::getSpeed() const {
        return _speed;
    }

    void Camera::update(double deltaSeconds) {
        const glm::vec3 dir = getDirection();
        const glm::vec3 up = getUp();
        const glm::vec3 side = getSide();

        // Update the position
        _position += dir  * _speed.z * (float)deltaSeconds;
        _position += up   * _speed.y * (float)deltaSeconds;
        _position += side * _speed.x * (float)deltaSeconds;

        _invalidateView();
    }

    const glm::mat4 & Camera::getViewTransform() const {
        _updateViewTransform();
        return _viewTransform;
    }

    const glm::mat4 & Camera::getWorldTransform() const {
        _updateViewTransform();
        return _worldTransform;
    }

    void Camera::_invalidateView() {
        _viewTransformValid = false;
    }

    void Camera::_updateViewTransform() const {
        if (_viewTransformValid) return;
        _worldTransform = constructTransformMat(_rotation, _position, glm::vec3(1.0f));
        _viewTransform = glm::inverse(_worldTransform);
        _viewTransformValid = true;
    }

// Camera::Camera(bool rangeCheckAngles) : _rangeCheckAngles(rangeCheckAngles) {
//     _side = glm::cross(_up, _dir);
//     _viewTransform = glm::lookAt(position, position + _dir, _up);
// }

// void Camera::modifyAngle(double deltaYaw, double deltaPitch) {
//     // rotation.x += deltaYaw;
//     // rotation.y += deltaPitch;
//     rotation.x -= deltaYaw;
//     rotation.y -= deltaPitch;

//     setAngle(rotation);
// }

// void Camera::setAngle(const glm::vec3 & angle) {
//     rotation = angle;
//     //if (_rangeCheckAngles) {
//     //    if (rotation.y > 89) rotation.y = 89.0f;
//     //    else if (rotation.y < -89.0f) rotation.y = -89.0f;
//     //}

//     // _dir = glm::normalize(
//     //     glm::vec3(cos(glm::radians(-getPitch())) * cos(glm::radians(getYaw())),
//     //             sin(glm::radians(-getPitch())),
//     //             cos(glm::radians(-getPitch())) * sin(glm::radians(getYaw()))));

//     _updateCameraAxes();
//     _updateViewTransform();

//     //_dir = glm::vec3(_worldTransform[2].x, _worldTransform[2].y, _worldTransform[2].z) * -1.0f;
//     //_dir = glm::vec3(_viewTransform[2]);
//     _dir = -glm::vec3(_worldTransform[2]);
// }

// void Camera::setSpeed(float forward, float up, float strafe) {
//     speed.z = forward;
//     speed.y = up;
//     speed.x = strafe;
// }

// const glm::vec3 &Camera::getSpeed() const {
//     return speed;
// }

// void Camera::update(double deltaSeconds) {
//     // Update the position
//     position += _dir * speed.z * (float)deltaSeconds;
//     position += _up * speed.y * (float)deltaSeconds;
//     position += _side * speed.x * (float)deltaSeconds;

//     // Update the view transform
//     _updateViewTransform();
//     //std::cout << "[cam pos] x: " << position.x <<
//     //    ", y: " << position.y << ", z: " << position.z << std::endl;
// }

// float Camera::getYaw() const {
//     return rotation.x;
// }

// float Camera::getPitch() const {
//     return rotation.y;
// }

// const glm::mat4 &Camera::getViewTransform() const {
//     return _viewTransform;
// }

// const glm::mat4& Camera::getWorldTransform() const {
//     return _worldTransform;
// }

// void Camera::setPosition(float x, float y, float z) {
//     Entity::setPosition(x, y, z);
// }

// const glm::vec3 &Camera::getPosition() const {
//     return position;
// }

// const glm::vec3 & Camera::getDirection() const {
//     return _dir;
// }

// void Camera::_updateViewTransform() {
//     // _viewTransform = glm::lookAt(position, position + _dir, _up);
//     //_viewTransform = constructViewMatrix(glm::vec3(getPitch(), getYaw(), rotation.z), position);
//     //_worldTransform = glm::inverse(_viewTransform);
//     _worldTransform = constructTransformMat(rotation, position, glm::vec3(1.0f));
//     _viewTransform = glm::inverse(_worldTransform);
//     //_viewTransform = glm::inverse(_worldTransform);
//     // See https://learnopengl.com/Getting-started/Camera for information about rotation + translation when
//     // creating view matrix. We use transpose of the rotation component since doing inverse results in precision errors.
//     // _viewTransform = glm::mat4(glm::mat3(glm::transpose(_worldTransform)));
//     // _viewTransform[3] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
//     // glm::mat4 translate = glm::mat4(1.0f);
//     // translate[3] = glm::vec4(-position, 1.0f);
//     // _viewTransform = _viewTransform * translate;
// }

// void Camera::_updateCameraAxes() {
//     // Update the camera's up and side to form an
//     // orthonormal basis
//     _up = glm::normalize(_worldUp - glm::dot(_worldUp, _dir) * _dir); // Gram-Schmidt
//     _side = glm::cross(_up, _dir);
// }
}