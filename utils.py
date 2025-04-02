import cv2
import numpy as np
import torch
from PIL import Image
import os
from typing import Tuple, Optional, Union, List
import logging
from pathlib import Path
from config import ModelConfig, OutputFormat
import open3d as o3d
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImageProcessingError(Exception):
    """Custom exception for image processing errors"""
    pass


class MeshProcessingError(Exception):
    """Custom exception for mesh processing errors"""
    pass


def preprocess_image(
    image_path: Union[str, np.ndarray],
    target_size: Tuple[int, int] = (224, 224),
    config: Optional[ModelConfig] = None
) -> torch.Tensor:
    """
    Preprocess image for model input with error handling.

    Args:
        image_path: Path to image or numpy array
        target_size: Target size for resizing
        config: Optional configuration object

    Returns:
        Preprocessed image tensor

    Raises:
        ImageProcessingError: If image processing fails
    """
    try:
        # Read image
        if isinstance(image_path, str):
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            image = cv2.imread(image_path)
            if image is None:
                raise ImageProcessingError(
                    f"Failed to read image: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_path

        # Validate image
        if image.size == 0:
            raise ImageProcessingError("Empty image")

        # Resize image
        image = cv2.resize(image, target_size)

        # Convert to PIL Image
        image = Image.fromarray(image)

        # Convert to tensor
        image = torch.from_numpy(np.array(image)).float() / 255.0
        image = image.permute(2, 0, 1).unsqueeze(0)

        return image

    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise ImageProcessingError(f"Failed to preprocess image: {str(e)}")


def save_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    texture: Optional[np.ndarray] = None,
    output_path: str = "model.obj",
    config: Optional[ModelConfig] = None
) -> None:
    """
    Save mesh to multiple formats with optional texture.

    Args:
        vertices: Vertex coordinates
        faces: Face indices
        texture: Optional texture map
        output_path: Output file path
        config: Optional configuration object

    Raises:
        MeshProcessingError: If mesh saving fails
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create Open3D mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.compute_vertex_normals()

        # Save in all requested formats
        for format in config.output_formats:
            format_path = output_path.with_suffix(f'.{format.value}')

            if format == OutputFormat.OBJ:
                save_obj(mesh, format_path, texture, config)
            elif format == OutputFormat.PLY:
                save_ply(mesh, format_path, texture, config)
            elif format == OutputFormat.STL:
                save_stl(mesh, format_path, config)
            elif format == OutputFormat.GLTF:
                save_gltf(mesh, format_path, texture, config)

        # Save texture if available
        if texture is not None:
            texture_path = output_path.with_suffix('.png')
            cv2.imwrite(str(texture_path), (texture * 255).astype(np.uint8))

    except Exception as e:
        logger.error(f"Error saving mesh: {str(e)}")
        raise MeshProcessingError(f"Failed to save mesh: {str(e)}")


def save_obj(mesh: o3d.geometry.TriangleMesh, path: Path, texture: Optional[np.ndarray], config: ModelConfig) -> None:
    """Save mesh in OBJ format."""
    with open(path, 'w') as f:
        # Write vertices
        for v in mesh.vertices:
            f.write(f'v {v[0]} {v[1]} {v[2]}\n')

        # Write texture coordinates if available
        if texture is not None:
            for i in range(len(mesh.vertices)):
                u = i % texture.shape[1] / texture.shape[1]
                v = i // texture.shape[1] / texture.shape[0]
                f.write(f'vt {u} {v}\n')

        # Write faces
        for face in mesh.triangles:
            if texture is not None:
                f.write(
                    f'f {face[0]+1}/{face[0]+1} {face[1]+1}/{face[1]+1} {face[2]+1}/{face[2]+1}\n')
            else:
                f.write(f'f {face[0]+1} {face[1]+1} {face[2]+1}\n')


def save_ply(mesh: o3d.geometry.TriangleMesh, path: Path, texture: Optional[np.ndarray], config: ModelConfig) -> None:
    """Save mesh in PLY format."""
    if config.output_ascii:
        o3d.io.write_triangle_mesh(str(path), mesh, write_ascii=True)
    else:
        o3d.io.write_triangle_mesh(
            str(path), mesh, write_ascii=False, compressed=config.compress_output)


def save_stl(mesh: o3d.geometry.TriangleMesh, path: Path, config: ModelConfig) -> None:
    """Save mesh in STL format."""
    if config.output_ascii:
        o3d.io.write_triangle_mesh(str(path), mesh, write_ascii=True)
    else:
        o3d.io.write_triangle_mesh(str(path), mesh, write_ascii=False)


def save_gltf(mesh: o3d.geometry.TriangleMesh, path: Path, texture: Optional[np.ndarray], config: ModelConfig) -> None:
    """Save mesh in GLTF format."""
    # Convert to GLTF format using Open3D
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    # Save as GLTF
    o3d.t.io.write_triangle_mesh(str(path), scene)


def visualize_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    texture: Optional[np.ndarray] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize mesh using matplotlib with optional texture.

    Args:
        vertices: Vertex coordinates
        faces: Face indices
        texture: Optional texture map
        save_path: Optional path to save visualization
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot vertices
        if texture is not None:
            colors = texture.reshape(-1, 3)
            ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                       c=colors, marker='o')
        else:
            ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                       c='b', marker='o')

        # Plot faces
        for face in faces:
            x = vertices[face, 0]
            y = vertices[face, 1]
            z = vertices[face, 2]
            ax.plot_trisurf(x, y, z)

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    except Exception as e:
        logger.error(f"Error visualizing mesh: {str(e)}")
        raise MeshProcessingError(f"Failed to visualize mesh: {str(e)}")


def ensure_dir(directory: Union[str, Path]) -> None:
    """
    Ensure directory exists, create if it doesn't.

    Args:
        directory: Directory path
    """
    try:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Error creating directory: {str(e)}")
        raise OSError(f"Failed to create directory: {str(e)}")


def validate_mesh(vertices: np.ndarray, faces: np.ndarray) -> bool:
    """
    Validate mesh geometry.

    Args:
        vertices: Vertex coordinates
        faces: Face indices

    Returns:
        bool: True if mesh is valid
    """
    try:
        # Check vertex count
        if len(vertices) == 0:
            return False

        # Check face indices
        if len(faces) == 0:
            return False

        # Check if face indices are valid
        max_vertex_idx = len(vertices) - 1
        for face in faces:
            if not all(0 <= idx <= max_vertex_idx for idx in face):
                return False

        return True

    except Exception as e:
        logger.error(f"Error validating mesh: {str(e)}")
        return False
