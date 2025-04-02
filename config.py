from dataclasses import dataclass
from typing import Tuple, Optional, List
from enum import Enum


class OutputFormat(Enum):
    OBJ = "obj"
    PLY = "ply"
    STL = "stl"
    GLTF = "gltf"


@dataclass
class ModelConfig:
    # Image processing
    image_size: Tuple[int, int] = (224, 224)
    input_channels: int = 3

    # CLIP model
    clip_model_name: str = "openai/clip-vit-base-patch32"

    # Encoder configuration
    encoder_channels: Tuple[int, ...] = (512, 256, 128, 64)
    encoder_kernel_size: int = 3

    # Decoder configuration
    decoder_channels: Tuple[int, ...] = (64, 32, 16, 3)
    decoder_kernel_size: int = 4
    decoder_stride: int = 2

    # Depth prediction
    depth_channels: Tuple[int, ...] = (64, 32, 1)
    depth_kernel_size: int = 3

    # Mesh generation
    marching_cubes_threshold: float = 0.5
    marching_cubes_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    marching_cubes_smooth: bool = True
    marching_cubes_smooth_iterations: int = 2
    mesh_resolution: int = 64
    min_vertices: int = 100
    max_vertices: int = 10000

    # Mesh simplification
    simplify_target_vertices: int = 5000
    simplify_target_faces: int = 10000
    simplify_quality_threshold: float = 0.1
    simplify_preserve_boundary: bool = True
    simplify_algorithm: str = "quadric"  # or "clustering"

    # Performance optimization
    use_parallel_processing: bool = True
    num_workers: int = 4
    chunk_size: int = 1000
    use_gpu_acceleration: bool = True
    batch_size: int = 32

    # Training parameters (if needed)
    learning_rate: float = 1e-4
    num_epochs: int = 100

    # Device configuration
    device: str = "cuda"  # or "cpu"

    # Output configuration
    output_formats: List[OutputFormat] = (
        OutputFormat.OBJ, OutputFormat.PLY, OutputFormat.STL)
    save_textures: bool = True
    texture_size: Tuple[int, int] = (1024, 1024)
    compress_output: bool = False
    output_ascii: bool = False  # For PLY/STL formats


# Default configuration
default_config = ModelConfig()
