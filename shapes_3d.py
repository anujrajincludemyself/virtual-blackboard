import numpy as np
import cv2

class Shape3D:
    """Base class for 3D shapes with transformation and rendering capabilities"""
    
    def __init__(self, position=(0, 0, 0), size=100):
        self.position = np.array(position, dtype=float)
        self.size = size
        self.rotation = np.array([0.0, 0.0, 0.0])  # Rotation angles in radians (X, Y, Z)
        self.vertices = []
        self.edges = []
        
    def rotate_x(self, angle):
        """Rotation matrix around X axis"""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])
    
    def rotate_y(self, angle):
        """Rotation matrix around Y axis"""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])
    
    def rotate_z(self, angle):
        """Rotation matrix around Z axis"""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
    
    def project_to_2d(self, point, distance=500):
        """Project 3D point to 2D using perspective projection"""
        x, y, z = point
        factor = distance / (distance + z)
        x_2d = int(x * factor + self.position[0])
        y_2d = int(y * factor + self.position[1])
        return (x_2d, y_2d)
    
    def get_transformed_vertices(self):
        """Apply rotation and scaling to vertices"""
        # Create rotation matrix
        rx = self.rotate_x(self.rotation[0])
        ry = self.rotate_y(self.rotation[1])
        rz = self.rotate_z(self.rotation[2])
        rotation_matrix = rz @ ry @ rx
        
        # Apply transformations
        transformed = []
        for vertex in self.vertices:
            # Scale
            scaled = vertex * self.size
            # Rotate
            rotated = rotation_matrix @ scaled
            transformed.append(rotated)
        
        return transformed
    
    def render(self, img, color=(255, 255, 255), thickness=2):
        """Render the 3D shape on the image"""
        transformed_vertices = self.get_transformed_vertices()
        projected_vertices = [self.project_to_2d(v) for v in transformed_vertices]
        
        # Draw edges
        for edge in self.edges:
            pt1 = projected_vertices[edge[0]]
            pt2 = projected_vertices[edge[1]]
            cv2.line(img, pt1, pt2, color, thickness)
        
        return img


class Cube(Shape3D):
    """3D Cube shape"""
    
    def __init__(self, position=(0, 0, 0), size=100):
        super().__init__(position, size)
        
        # Define cube vertices (normalized to unit cube)
        self.vertices = [
            np.array([-0.5, -0.5, -0.5]),
            np.array([0.5, -0.5, -0.5]),
            np.array([0.5, 0.5, -0.5]),
            np.array([-0.5, 0.5, -0.5]),
            np.array([-0.5, -0.5, 0.5]),
            np.array([0.5, -0.5, 0.5]),
            np.array([0.5, 0.5, 0.5]),
            np.array([-0.5, 0.5, 0.5])
        ]
        
        # Define edges (pairs of vertex indices)
        self.edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Back face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Front face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges
        ]


class Sphere(Shape3D):
    """3D Sphere shape (wireframe approximation)"""
    
    def __init__(self, position=(0, 0, 0), size=100, segments=12):
        super().__init__(position, size)
        self.segments = segments
        
        # Create vertices for latitude and longitude lines
        self.vertices = []
        self.edges = []
        
        # Latitude circles
        for i in range(segments):
            theta = (i / segments) * np.pi  # 0 to π
            for j in range(segments):
                phi = (j / segments) * 2 * np.pi  # 0 to 2π
                x = 0.5 * np.sin(theta) * np.cos(phi)
                y = 0.5 * np.sin(theta) * np.sin(phi)
                z = 0.5 * np.cos(theta)
                self.vertices.append(np.array([x, y, z]))
        
        # Create edges for wireframe
        for i in range(segments):
            for j in range(segments):
                current = i * segments + j
                next_j = i * segments + ((j + 1) % segments)
                if i < segments - 1:
                    next_i = (i + 1) * segments + j
                    self.edges.append((current, next_i))
                self.edges.append((current, next_j))


class Pyramid(Shape3D):
    """3D Pyramid shape"""
    
    def __init__(self, position=(0, 0, 0), size=100):
        super().__init__(position, size)
        
        # Define pyramid vertices
        self.vertices = [
            np.array([0, -0.5, 0]),      # Apex
            np.array([-0.5, 0.5, -0.5]), # Base corners
            np.array([0.5, 0.5, -0.5]),
            np.array([0.5, 0.5, 0.5]),
            np.array([-0.5, 0.5, 0.5])
        ]
        
        # Define edges
        self.edges = [
            (0, 1), (0, 2), (0, 3), (0, 4),  # Apex to base
            (1, 2), (2, 3), (3, 4), (4, 1)   # Base edges
        ]

