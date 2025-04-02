import argparse
import torch
import logging
from pathlib import Path
from typing import Optional
import json

from model import Zero123Plus
from utils import (
    preprocess_image, save_mesh, visualize_mesh, ensure_dir,
    ImageProcessingError, MeshProcessingError, validate_mesh
)
from config import ModelConfig, default_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: Optional[str] = None) -> ModelConfig:
    """Load configuration from file or use default."""
    if config_path:
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            return ModelConfig(**config_dict)
        except Exception as e:
            logger.warning(
                f"Failed to load config file: {str(e)}. Using default config.")
    return default_config


def main():
    parser = argparse.ArgumentParser(
        description='Convert 2D image to 3D model')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to save output 3D model')
    parser.add_argument('--config', type=str,
                        help='Path to configuration file')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the generated mesh')
    parser.add_argument('--save-visualization', type=str,
                        help='Path to save visualization')
    parser.add_argument('--device', type=str,
                        choices=['cuda', 'cpu'], help='Device to run on')
    parser.add_argument('--image-size', type=int, nargs=2,
                        help='Target image size (width height)')
    parser.add_argument('--mesh-resolution', type=int,
                        help='Mesh resolution for marching cubes')
    parser.add_argument('--save-texture', action='store_true',
                        help='Save texture map')

    # Marching cubes options
    parser.add_argument('--mc-threshold', type=float,
                        help='Marching cubes threshold')
    parser.add_argument('--mc-spacing', type=float, nargs=3,
                        help='Marching cubes spacing (x y z)')
    parser.add_argument('--mc-smooth', action='store_true',
                        help='Enable marching cubes smoothing')
    parser.add_argument('--mc-smooth-iterations', type=int,
                        help='Number of smoothing iterations')

    # Mesh simplification options
    parser.add_argument('--simplify-target-vertices', type=int,
                        help='Target number of vertices after simplification')
    parser.add_argument('--simplify-target-faces', type=int,
                        help='Target number of faces after simplification')
    parser.add_argument('--simplify-quality-threshold', type=float,
                        help='Quality threshold for mesh simplification')
    parser.add_argument('--simplify-preserve-boundary', action='store_true',
                        help='Preserve mesh boundaries during simplification')
    parser.add_argument('--simplify-algorithm',
                        choices=['quadric', 'clustering'], help='Mesh simplification algorithm')

    args = parser.parse_args()

    try:
        # Load configuration
        config = load_config(args.config)

        # Override config with command line arguments
        if args.device:
            config.device = args.device
        if args.image_size:
            config.image_size = tuple(args.image_size)
        if args.mesh_resolution:
            config.mesh_resolution = args.mesh_resolution

        # Marching cubes options
        if args.mc_threshold is not None:
            config.marching_cubes_threshold = args.mc_threshold
        if args.mc_spacing:
            config.marching_cubes_spacing = tuple(args.mc_spacing)
        if args.mc_smooth:
            config.marching_cubes_smooth = True
        if args.mc_smooth_iterations:
            config.marching_cubes_smooth_iterations = args.mc_smooth_iterations

        # Mesh simplification options
        if args.simplify_target_vertices:
            config.simplify_target_vertices = args.simplify_target_vertices
        if args.simplify_target_faces:
            config.simplify_target_faces = args.simplify_target_faces
        if args.simplify_quality_threshold:
            config.simplify_quality_threshold = args.simplify_quality_threshold
        if args.simplify_preserve_boundary:
            config.simplify_preserve_boundary = True
        if args.simplify_algorithm:
            config.simplify_algorithm = args.simplify_algorithm

        logger.info(f"Using configuration: {config}")

        # Ensure output directory exists
        ensure_dir(args.output)

        # Set device
        device = torch.device(config.device)
        logger.info(f"Using device: {device}")

        # Load and preprocess image
        logger.info('Loading and preprocessing image...')
        try:
            image = preprocess_image(args.input, config.image_size, config)
            image = image.to(device)
        except ImageProcessingError as e:
            logger.error(f"Failed to process image: {str(e)}")
            return

        # Initialize model
        logger.info('Initializing model...')
        model = Zero123Plus(config)
        model.to(device)

        # Generate 3D mesh
        logger.info('Generating 3D mesh...')
        try:
            vertices, faces, texture = model.generate_mesh(image)
        except Exception as e:
            logger.error(f"Failed to generate mesh: {str(e)}")
            return

        # Validate mesh
        if not validate_mesh(vertices, faces):
            logger.error("Generated mesh is invalid")
            return

        # Save mesh
        output_path = Path(args.output) / "model.obj"
        logger.info(f'Saving mesh to {output_path}...')
        try:
            save_mesh(
                vertices, faces,
                texture if args.save_texture else None,
                str(output_path),
                config
            )
        except MeshProcessingError as e:
            logger.error(f"Failed to save mesh: {str(e)}")
            return

        # Visualize if requested
        if args.visualize or args.save_visualization:
            logger.info('Visualizing mesh...')
            try:
                visualize_mesh(
                    vertices, faces,
                    texture if args.save_texture else None,
                    args.save_visualization
                )
            except MeshProcessingError as e:
                logger.error(f"Failed to visualize mesh: {str(e)}")
                return

        logger.info('Done!')

    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        return


if __name__ == '__main__':
    main()
