# Video Frame Reconstruction

A Python toolkit for reconstructing jumbled video frames back into their correct temporal sequence using computer vision and optimization algorithms.

## Features

- **Two Reconstruction Algorithms**:
  - Basic greedy and bidirectional reconstruction
  - Enhanced reconstruction with lookahead and 2-opt refinement
- **Multiple Similarity Metrics**: Histogram correlation, MSE, edge detection, SSIM
- **Evaluation Tools**: Compare reconstructed videos with originals, analyze temporal consistency
- **Progress Tracking**: Real-time progress bars and execution logging

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Reconstruction
```bash
python reconstruct_video.py input_video.mp4 -o output.mp4 -m greedy
```

### Enhanced Reconstruction (Recommended)
```bash
python enhanced_reconstructor.py input_video.mp4 -o output.mp4
```

### Evaluation
```bash
# Compare with original
python test_reconstruction.py -r reconstructed.mp4 -o original.mp4 -m compare

# Check temporal consistency
python test_reconstruction.py -r reconstructed.mp4 -m temporal

# Full analysis
python test_reconstruction.py -r reconstructed.mp4 -o original.mp4 -m all
```

## Files

- `reconstruct_video.py` - Basic reconstruction with greedy/bidirectional algorithms
- `enhanced_reconstructor.py` - Advanced reconstruction with lookahead and refinement
- `test_reconstruction.py` - Evaluation and analysis tools
- `requirements.txt` - Python dependencies

## How It Works

1. **Load Frames**: Extract all frames from the jumbled video
2. **Compute Similarities**: Calculate pairwise frame similarities using multiple metrics
3. **Find Optimal Path**: Use optimization algorithms to find the best frame ordering
4. **Save Video**: Write frames in reconstructed order to output file

## Requirements

- Python 3.7+
- OpenCV
- NumPy
- scikit-learn
- matplotlib
- tqdm