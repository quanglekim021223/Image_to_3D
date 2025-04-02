# Image to 3D Model Generator

This project implements a state-of-the-art AI model for converting 2D images into high-quality 3D models. The implementation is based on Zero123Plus, a powerful model developed by Google Research.

## Features

- Convert single 2D image to 3D model
- High-quality mesh generation
- Support for various input image formats
- Optimized for Google Colab/Kaggle free tier
- Export to common 3D formats (OBJ, PLY)

## Setup

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your input image (preferably front view of the object)
2. Run the conversion script:
```bash
python convert.py --input path/to/your/image.jpg --output path/to/save/model
```

## Project Structure

- `convert.py`: Main conversion script
- `model.py`: Zero123Plus model implementation
- `utils.py`: Utility functions
- `requirements.txt`: Project dependencies

## Technical Details

This implementation uses:
- Zero123Plus model architecture
- PyTorch for deep learning
- Open3D for 3D processing
- Transformers for feature extraction

## License

MIT License

## Acknowledgments

- Zero123Plus paper and implementation
- Google Research team
- Open source community 