#version 150 core

in vec3 fsPosition;
in vec3 fsNormal;
in vec3 fsTexCoords;

uniform vec3 diffuseColor;
uniform vec3 ambientColor;
uniform vec3 specularColor;
uniform float shininess = 0.0;
uniform sampler2D diffuseTexture;

uniform vec3 lightPosition;
uniform vec3 lightColor;

vec3 calculatePointLighting(vec3 baseColor, vec3 viewDir, int lightIndex) {
    vec3 lightDir = lightPosition - fsPosition;
}