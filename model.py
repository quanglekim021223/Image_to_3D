import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from skimage import measure
from skimage.filters import gaussian
from config import ModelConfig, OutputFormat
import open3d as o3d
from typing import Tuple, Optional, List
import concurrent.futures
from functools import partial
import multiprocessing
from tqdm import tqdm


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)


class Zero123Plus(nn.Module):
    def __init__(self, config: ModelConfig = None):
        super().__init__()
        self.config = config or ModelConfig()
        self.device = torch.device(self.config.device)

        # Load CLIP model for feature extraction
        self.clip_model = CLIPModel.from_pretrained(
            self.config.clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(
            self.config.clip_model_name)

        # Improved encoder with residual blocks
        self.encoder = nn.ModuleList()
        in_channels = 512
        for out_channels in self.config.encoder_channels:
            self.encoder.append(ResidualBlock(in_channels, out_channels))
            in_channels = out_channels

        # Improved decoder with skip connections
        self.decoder = nn.ModuleList()
        in_channels = self.config.encoder_channels[-1]
        for out_channels in self.config.decoder_channels:
            self.decoder.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels,
                                   self.config.decoder_kernel_size,
                                   stride=self.config.decoder_stride,
                                   padding=self.config.decoder_stride//2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ))
            in_channels = out_channels

        # Enhanced depth prediction network
        self.depth_net = nn.Sequential(
            ResidualBlock(self.config.encoder_channels[-1], 64),
            nn.Conv2d(64, 1, self.config.depth_kernel_size, padding=1),
            nn.Sigmoid()
        )

        # Texture prediction network
        self.texture_net = nn.Sequential(
            ResidualBlock(self.config.encoder_channels[-1], 64),
            nn.Conv2d(64, 3, self.config.depth_kernel_size, padding=1),
            nn.Sigmoid()
        )

        # Initialize parallel processing pool if needed
        if self.config.use_parallel_processing:
            self.process_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.config.num_workers
            )

    def extract_features(self, image):
        inputs = self.clip_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)

        return image_features

    def forward(self, image):
        # Extract features using CLIP
        features = self.extract_features(image)
        features = features.view(-1, 512, 8, 8)

        # Encoder path with skip connections
        skip_connections = []
        x = features
        for encoder_block in self.encoder:
            x = encoder_block(x)
            skip_connections.append(x)

        # Decoder path with skip connections
        for i, decoder_block in enumerate(self.decoder):
            x = decoder_block(x)
            if i < len(skip_connections) - 1:  # Skip the last skip connection
                x = x + skip_connections[-(i+2)]

        # Predict depth and texture
        depth = self.depth_net(skip_connections[-1])
        texture = self.texture_net(skip_connections[-1])

        return x, depth, texture

    def preprocess_depth(self, depth: np.ndarray) -> np.ndarray:
        """Preprocess depth map for marching cubes."""
        # Apply Gaussian smoothing if enabled
        if self.config.marching_cubes_smooth:
            depth = gaussian(
                depth, sigma=1.0, iterations=self.config.marching_cubes_smooth_iterations)

        # Normalize depth values
        depth = (depth - depth.min()) / (depth.max() - depth.min())

        return depth

    def process_chunk(self, chunk_vertices: np.ndarray, chunk_faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Process a chunk of the mesh in parallel."""
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(chunk_vertices)
        mesh.triangles = o3d.utility.Vector3iVector(chunk_faces)
        mesh.compute_vertex_normals()

        if self.config.simplify_algorithm == "quadric":
            mesh = mesh.simplify_quadric_decimation(
                target_number_of_triangles=self.config.simplify_target_faces // self.config.num_workers,
                quality_threshold=self.config.simplify_quality_threshold,
                preserve_boundary=self.config.simplify_preserve_boundary
            )
        else:
            mesh = mesh.simplify_vertex_clustering(
                voxel_size=0.05,
                contraction=o3d.geometry.SimplificationContraction.Average
            )

        return np.asarray(mesh.vertices), np.asarray(mesh.triangles)

    def simplify_mesh(self, vertices: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simplify mesh using parallel processing and Open3D's mesh simplification algorithms.
        """
        if not self.config.use_parallel_processing:
            # Use single-threaded processing for small meshes
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            mesh.compute_vertex_normals()

            if self.config.simplify_algorithm == "quadric":
                mesh = mesh.simplify_quadric_decimation(
                    target_number_of_triangles=self.config.simplify_target_faces,
                    quality_threshold=self.config.simplify_quality_threshold,
                    preserve_boundary=self.config.simplify_preserve_boundary
                )
            else:
                mesh = mesh.simplify_vertex_clustering(
                    voxel_size=0.05,
                    contraction=o3d.geometry.SimplificationContraction.Average
                )

            return np.asarray(mesh.vertices), np.asarray(mesh.triangles)

        # Parallel processing for large meshes
        num_chunks = max(1, len(faces) // self.config.chunk_size)
        chunk_size = len(faces) // num_chunks

        chunks = []
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = start_idx + \
                chunk_size if i < num_chunks - 1 else len(faces)
            chunk_faces = faces[start_idx:end_idx]
            chunk_vertices = vertices[list(set(chunk_faces.flatten()))]
            chunks.append((chunk_vertices, chunk_faces))

        # Process chunks in parallel
        with tqdm(total=len(chunks), desc="Simplifying mesh chunks") as pbar:
            futures = []
            for chunk in chunks:
                future = self.process_pool.submit(self.process_chunk, *chunk)
                futures.append(future)

            results = []
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                pbar.update(1)

        # Merge results
        merged_vertices = np.vstack([r[0] for r in results])
        merged_faces = np.vstack([r[1] for r in results])

        # Final simplification if needed
        if len(merged_vertices) > self.config.simplify_target_vertices:
            final_mesh = o3d.geometry.TriangleMesh()
            final_mesh.vertices = o3d.utility.Vector3dVector(merged_vertices)
            final_mesh.triangles = o3d.utility.Vector3iVector(merged_faces)
            final_mesh.compute_vertex_normals()

            final_mesh = final_mesh.simplify_quadric_decimation(
                target_number_of_triangles=self.config.simplify_target_vertices * 2,
                quality_threshold=self.config.simplify_quality_threshold,
                preserve_boundary=self.config.simplify_preserve_boundary
            )

            return np.asarray(final_mesh.vertices), np.asarray(final_mesh.triangles)

        return merged_vertices, merged_faces

    def generate_mesh(self, image):
        self.eval()
        with torch.no_grad():
            decoded, depth, texture = self(image)

        # Convert to numpy arrays
        depth = depth.cpu().numpy()[0, 0]
        texture = texture.cpu().numpy()[0]

        # Preprocess depth map
        depth = self.preprocess_depth(depth)

        # Apply marching cubes
        vertices, faces, normals, values = measure.marching_cubes(
            depth,
            level=self.config.marching_cubes_threshold,
            spacing=self.config.marching_cubes_spacing
        )

        # Filter vertices based on configuration
        if len(vertices) < self.config.min_vertices:
            raise ValueError(
                f"Generated mesh has too few vertices: {len(vertices)}")
        if len(vertices) > self.config.max_vertices:
            # Simplify mesh if needed
            vertices, faces = self.simplify_mesh(vertices, faces)

        return vertices, faces, texture
