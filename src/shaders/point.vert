// Uniform variables
uniform vec3 cameraWorldPosition;           // World position of the camera
uniform mat4 modelToWorldMatrix;        // Model too world coordinates matrix
uniform mat4 worldToCameraMatrix;       // World to camera coordinates matrix
uniform mat4 projectionMatrix;          // Projection matrix

// Varying variables
varying vec3 worldPosition;             // World position of the vertex

void main() {

    // Compute the vertex position
    vec4 worldPos = modelToWorldMatrix * gl_Vertex;
    worldPosition = worldPos.xyz;

    // Compute the clip-space vertex coordinates
    gl_Position = projectionMatrix * worldToCameraMatrix * worldPos;
}
