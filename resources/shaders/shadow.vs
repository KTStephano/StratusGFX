#version 400 core

//#define MAX_INSTANCES 250

layout (location = 0) in vec3 position;
layout (location = 12) in mat4 model;

//uniform mat4 modelMats[MAX_INSTANCES];

void main() {
    //mat4 model = modelMats[gl_InstanceID];
    gl_Position = model * vec4(position, 1.0);
}