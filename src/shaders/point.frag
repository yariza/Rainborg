// Uniform variables
uniform vec3 cameraWorldPosition;           // World position of the camera

// Varying variables
varying vec3 worldPosition;             // World position of the vertex

void main() {

    // Compute the final color
    gl_FragColor = vec4(1.0, 1.0, 1.0, 0.6);
}
