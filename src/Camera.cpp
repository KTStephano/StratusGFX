
#include <Camera.h>
#include <iostream>

#include "Camera.h"
#include "Utils.h"

namespace stratus {
Camera::Camera() {
    _side = glm::cross(_up, _dir);
    _viewTransform = glm::lookAt(position, position + _dir, _up);
}

void Camera::modifyAngle(double deltaYaw, double deltaPitch) {
    // rotation.x += deltaYaw;
    // rotation.y += deltaPitch;
    rotation.x -= deltaYaw;
    rotation.y -= deltaPitch;

    setAngle(glm::vec3(rotation.x, rotation.y, 0.0f));
}

void Camera::setAngle(const glm::vec3 & angle) {
    rotation = angle;
    if (rotation.y > 89) rotation.y = 89.0f;
    else if (rotation.y < -89.0f) rotation.y = -89.0f;

    // _dir = glm::normalize(
    //     glm::vec3(cos(glm::radians(-getPitch())) * cos(glm::radians(getYaw())),
    //             sin(glm::radians(-getPitch())),
    //             cos(glm::radians(-getPitch())) * sin(glm::radians(getYaw()))));

    _updateCameraAxes();
    _updateViewTransform();

    _dir = glm::vec3(_worldTransform[2].x, _worldTransform[2].y, _worldTransform[2].z) * -1.0f;
}

void Camera::setSpeed(float forward, float up, float strafe) {
    speed.z = forward;
    speed.y = up;
    speed.x = strafe;
}

const glm::vec3 &Camera::getSpeed() const {
    return speed;
}

void Camera::update(double deltaSeconds) {
    // Update the position
    position += _dir * speed.z * (float)deltaSeconds;
    position += _up * speed.y * (float)deltaSeconds;
    position += _side * speed.x * (float)deltaSeconds;

    // Update the view transform
    _updateViewTransform();
    //std::cout << "[cam pos] x: " << position.x <<
    //    ", y: " << position.y << ", z: " << position.z << std::endl;
}

float Camera::getYaw() const {
    return rotation.x;
}

float Camera::getPitch() const {
    return rotation.y;
}

const glm::mat4 &Camera::getViewTransform() const {
    return _viewTransform;
}

void Camera::setPosition(float x, float y, float z) {
    Entity::setPosition(x, y, z);
}

const glm::vec3 &Camera::getPosition() const {
    return position;
}

const glm::vec3 & Camera::getDirection() const {
    return _dir;
}

void Camera::_updateViewTransform() {
    // _viewTransform = glm::lookAt(position, position + _dir, _up);
    // std::cout << "First version " << std::endl << _viewTransform << std::endl << std::endl;
    _viewTransform = constructViewMatrix(glm::vec3(getPitch(), getYaw(), 0.0f), position);
    _worldTransform = glm::inverse(_viewTransform);
    // std::cout << "Second version" << std::endl << _viewTransform << std::endl << std::endl;
}

void Camera::_updateCameraAxes() {
    // Update the camera's up and side to form an
    // orthonormal basis
    _up = glm::normalize(_worldUp - glm::dot(_worldUp, _dir) * _dir); // Gram-Schmidt
    _side = glm::cross(_up, _dir);
}
}