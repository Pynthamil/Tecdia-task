import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

class ReconstructionEvaluator:
    def __init__(self, original_path=None, reconstructed_path=None):
        self.original_path = Path(original_path) if original_path else None
        self.reconstructed_path = Path(reconstructed_path) if reconstructed_path else None
        
    def load_video_frames(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        return frames
    
    def compute_frame_similarity(self, frame1, frame2):
        f1 = cv2.resize(frame1, (320, 180))
        f2 = cv2.resize(frame2, (320, 180))
        
        hist1 = cv2.calcHist([f1], [0, 1, 2], None, [8, 8, 8], 
                            [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([f2], [0, 1, 2], None, [8, 8, 8], 
                            [0, 256, 0, 256, 0, 256])
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()
        hist_sim = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        mse = np.mean((f1.astype(float) - f2.astype(float)) ** 2)
        mse_sim = 1.0 / (1.0 + mse / 1000.0)
        
        return 0.6 * hist_sim + 0.4 * mse_sim
    
    def evaluate_with_original(self):
        if not self.original_path or not self.reconstructed_path:
            raise ValueError("Both original and reconstructed paths required")
        
        print("Loading original video...")
        original_frames = self.load_video_frames(self.original_path)
        
        print("Loading reconstructed video...")
        reconstructed_frames = self.load_video_frames(self.reconstructed_path)
        
        n = len(original_frames)
        
        if len(reconstructed_frames) != n:
            print(f"Warning: Frame count mismatch! Original: {n}, Reconstructed: {len(reconstructed_frames)}")
            n = min(n, len(reconstructed_frames))
        
        print(f"\nEvaluating {n} frames...")
        
        similarities = []
        for i in tqdm(range(n), desc="Computing similarities"):
            sim = self.compute_frame_similarity(original_frames[i], 
                                               reconstructed_frames[i])
            similarities.append(sim)
        
        avg_similarity = np.mean(similarities)
        min_similarity = np.min(similarities)
        max_similarity = np.max(similarities)
        std_similarity = np.std(similarities)
        
        exact_matches = sum(1 for s in similarities if s > 0.95)
        
        results = {
            'total_frames': n,
            'average_similarity': float(avg_similarity),
            'min_similarity': float(min_similarity),
            'max_similarity': float(max_similarity),
            'std_similarity': float(std_similarity),
            'exact_matches': exact_matches,
            'exact_match_percentage': 100 * exact_matches / n,
            'frame_similarities': [float(s) for s in similarities]
        }
        
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Total Frames:           {n}")
        print(f"Average Similarity:     {avg_similarity:.4f} ({avg_similarity*100:.2f}%)")
        print(f"Min Similarity:         {min_similarity:.4f}")
        print(f"Max Similarity:         {max_similarity:.4f}")
        print(f"Std Deviation:          {std_similarity:.4f}")
        print(f"Exact Matches (>0.95):  {exact_matches} ({100*exact_matches/n:.2f}%)")
        print("="*60)
        
        with open('evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        self.plot_similarity_graph(similarities)
        
        return results
    
    def evaluate_temporal_consistency(self):
        if not self.reconstructed_path:
            raise ValueError("Reconstructed path required")
        
        print("Loading reconstructed video...")
        frames = self.load_video_frames(self.reconstructed_path)
        n = len(frames)
        
        print(f"\nEvaluating temporal consistency for {n} frames...")
        
        consecutive_similarities = []
        for i in tqdm(range(n - 1), desc="Computing consecutive similarities"):
            sim = self.compute_frame_similarity(frames[i], frames[i + 1])
            consecutive_similarities.append(sim)
        
        avg_consecutive = np.mean(consecutive_similarities)
        min_consecutive = np.min(consecutive_similarities)
        std_consecutive = np.std(consecutive_similarities)
        
        discontinuities = [i for i, s in enumerate(consecutive_similarities) if s < 0.7]
        
        results = {
            'total_frames': n,
            'avg_consecutive_similarity': float(avg_consecutive),
            'min_consecutive_similarity': float(min_consecutive),
            'std_consecutive_similarity': float(std_consecutive),
            'num_discontinuities': len(discontinuities),
            'discontinuity_frames': discontinuities,
            'consecutive_similarities': [float(s) for s in consecutive_similarities]
        }
        
        print("\n" + "="*60)
        print("TEMPORAL CONSISTENCY RESULTS")
        print("="*60)
        print(f"Total Frames:                    {n}")
        print(f"Avg Consecutive Similarity:      {avg_consecutive:.4f} ({avg_consecutive*100:.2f}%)")
        print(f"Min Consecutive Similarity:      {min_consecutive:.4f}")
        print(f"Std Deviation:                   {std_consecutive:.4f}")
        print(f"Discontinuities (sim < 0.7):     {len(discontinuities)}")
        if discontinuities:
            print(f"Discontinuity Frames:            {discontinuities[:10]}{'...' if len(discontinuities) > 10 else ''}")
        print("="*60)
        
        with open('temporal_consistency_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        self.plot_temporal_consistency(consecutive_similarities)
        
        return results
    
    def plot_similarity_graph(self, similarities):
        plt.figure(figsize=(12, 6))
        plt.plot(similarities, linewidth=1, alpha=0.7)
        plt.axhline(y=np.mean(similarities), color='r', linestyle='--', 
                   label=f'Average: {np.mean(similarities):.4f}')
        plt.axhline(y=0.95, color='g', linestyle='--', alpha=0.5, 
                   label='Exact Match Threshold (0.95)')
        plt.xlabel('Frame Index')
        plt.ylabel('Similarity Score')
        plt.title('Frame-wise Similarity: Reconstructed vs Original')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1.05])
        plt.tight_layout()
        plt.savefig('similarity_graph.png', dpi=150)
        print("Similarity graph saved to: similarity_graph.png")
        plt.close()
    
    def plot_temporal_consistency(self, similarities):
        plt.figure(figsize=(12, 6))
        plt.plot(similarities, linewidth=1, alpha=0.7)
        plt.axhline(y=np.mean(similarities), color='r', linestyle='--', 
                   label=f'Average: {np.mean(similarities):.4f}')
        plt.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, 
                   label='Discontinuity Threshold (0.7)')
        plt.xlabel('Frame Transition Index')
        plt.ylabel('Similarity Score')
        plt.title('Temporal Consistency: Consecutive Frame Similarities')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1.05])
        plt.tight_layout()
        plt.savefig('temporal_consistency_graph.png', dpi=150)
        print("Temporal consistency graph saved to: temporal_consistency_graph.png")
        plt.close()
    
    def analyze_video_quality(self):
        if not self.reconstructed_path:
            raise ValueError("Reconstructed path required")
        
        print("Analyzing video quality...")
        frames = self.load_video_frames(self.reconstructed_path)
        
        brightness_values = []
        contrast_values = []
        sharpness_values = []
        
        for frame in tqdm(frames[:50], desc="Analyzing sample frames"):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            brightness = np.mean(gray)
            brightness_values.append(brightness)
            
            contrast = np.std(gray)
            contrast_values.append(contrast)
            
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()
            sharpness_values.append(sharpness)
        
        print("\nQuality Metrics (sample of 50 frames):")
        print(f"  Avg Brightness: {np.mean(brightness_values):.2f}")
        print(f"  Avg Contrast:   {np.mean(contrast_values):.2f}")
        print(f"  Avg Sharpness:  {np.mean(sharpness_values):.2f}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate video reconstruction')
    parser.add_argument('-o', '--original', help='Path to original video (optional)')
    parser.add_argument('-r', '--reconstructed', required=True,
                       help='Path to reconstructed video')
    parser.add_argument('-m', '--mode', choices=['compare', 'temporal', 'quality', 'all'],
                       default='all', help='Evaluation mode')
    
    args = parser.parse_args()
    
    evaluator = ReconstructionEvaluator(args.original, args.reconstructed)
    
    if args.mode in ['compare', 'all'] and args.original:
        print("\n=== COMPARING WITH ORIGINAL ===")
        evaluator.evaluate_with_original()
    
    if args.mode in ['temporal', 'all']:
        print("\n=== TEMPORAL CONSISTENCY ANALYSIS ===")
        evaluator.evaluate_temporal_consistency()
    
    if args.mode in ['quality', 'all']:
        print("\n=== QUALITY ANALYSIS ===")
        evaluator.analyze_video_quality()
    
    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
