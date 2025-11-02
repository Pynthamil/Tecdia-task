import cv2
import numpy as np
from pathlib import Path
import time
import json
from tqdm import tqdm
import argparse

class VideoReconstructor:
    def __init__(self, video_path):
        self.video_path = Path(video_path)
        self.frames = []
        self.frame_indices = []
        
    def load_frames(self):
        print("Loading frames from video...")
        cap = cv2.VideoCapture(str(self.video_path))
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video properties: {frame_count} frames, {fps} FPS, {width}x{height}")
        
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            resized = cv2.resize(frame, (640, 360))
            self.frames.append(resized)
            self.frame_indices.append(idx)
            idx += 1
        
        cap.release()
        print(f"Loaded {len(self.frames)} frames")
        return fps, width, height
    
    def compute_frame_similarity(self, idx1, idx2):
        frame1 = self.frames[idx1]
        frame2 = self.frames[idx2]
        
        hist1 = cv2.calcHist([frame1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([frame2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()
        hist_similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        mse = np.mean((frame1.astype(float) - frame2.astype(float)) ** 2)
        mse_similarity = 1.0 / (1.0 + mse / 1000.0)
        
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        edges1 = cv2.Canny(gray1, 50, 150)
        edges2 = cv2.Canny(gray2, 50, 150)
        edge_diff = np.sum(np.abs(edges1.astype(float) - edges2.astype(float)))
        edge_similarity = 1.0 / (1.0 + edge_diff / 10000.0)
        
        similarity = 0.4 * hist_similarity + 0.3 * mse_similarity + 0.3 * edge_similarity
        
        return similarity
    
    def build_similarity_matrix(self):
        n = len(self.frames)
        print(f"\nBuilding similarity matrix for {n} frames...")
        
        similarity_matrix = np.zeros((n, n))
        
        print("Computing frame similarities...")
        for i in tqdm(range(n), desc="Processing frames"):
            for j in range(i + 1, n):
                sim = self.compute_frame_similarity(i, j)
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
        
        np.fill_diagonal(similarity_matrix, 1.0)
        
        return similarity_matrix
    
    def greedy_path_reconstruction(self, similarity_matrix):
        print("\nReconstructing frame order using greedy algorithm...")
        n = len(similarity_matrix)
        
        if n == 0:
            print("Error: Empty similarity matrix!")
            return None
        
        if n == 1:
            print("Only one frame, returning trivial path")
            return [0]
        
        best_path = None
        best_score = -np.inf
        
        num_trials = min(5, n)
        start_candidates = np.linspace(0, n-1, num_trials, dtype=int)
        
        for start_idx in tqdm(start_candidates, desc="Trying start positions"):
            path = [start_idx]
            visited = {start_idx}
            
            while len(visited) < n:
                current = path[-1]
                
                similarities = similarity_matrix[current].copy()
                similarities[list(visited)] = -np.inf
                next_frame = np.argmax(similarities)
                
                path.append(next_frame)
                visited.add(next_frame)
            
            score = sum(similarity_matrix[path[i], path[i+1]] for i in range(len(path)-1))
            
            if score > best_score:
                best_score = score
                best_path = path
        
        if best_path is None:
            print("Warning: Could not find valid path, using sequential order")
            best_path = list(range(n))
        
        print(f"Best path score: {best_score:.2f}")
        return best_path
    
    def bidirectional_reconstruction(self, similarity_matrix):
        print("\nAttempting bidirectional reconstruction...")
        n = len(similarity_matrix)
        
        avg_similarities = np.mean(similarity_matrix, axis=1)
        potential_endpoints = np.argsort(avg_similarities)[:10]
        
        best_path = None
        best_score = -np.inf
        
        for start_idx in tqdm(potential_endpoints, desc="Trying endpoints"):
            forward_path = [start_idx]
            visited = {start_idx}
            
            while len(visited) < n:
                current = forward_path[-1]
                similarities = similarity_matrix[current].copy()
                similarities[list(visited)] = -np.inf
                next_frame = np.argmax(similarities)
                
                if similarities[next_frame] < 0.5:
                    break
                    
                forward_path.append(next_frame)
                visited.add(next_frame)
            
            if len(forward_path) == n:
                score = sum(similarity_matrix[forward_path[i], forward_path[i+1]] 
                           for i in range(len(forward_path)-1))
                
                if score > best_score:
                    best_score = score
                    best_path = forward_path
        
        return best_path if best_path else self.greedy_path_reconstruction(similarity_matrix)
    
    def save_reconstructed_video(self, frame_order, output_path, original_fps, original_width, original_height):
        print(f"\nSaving reconstructed video to {output_path}...")
        
        cap = cv2.VideoCapture(str(self.video_path))
        original_frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            original_frames.append(frame)
        cap.release()
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, original_fps, 
                             (original_width, original_height))
        
        if not out.isOpened():
            print(f"Error: Could not open video writer for {output_path}")
            return
        
        for idx in tqdm(frame_order, desc="Writing frames"):
            out.write(original_frames[idx])
        
        out.release()
        print(f"Video saved successfully!")
    
    def reconstruct(self, output_path='reconstructed_video.mp4', method='greedy'):
        start_time = time.time()
        
        fps, width, height = self.load_frames()
        
        if len(self.frames) == 0:
            print("Error: No frames loaded from video!")
            return None, 0
        
        similarity_matrix = self.build_similarity_matrix()
        
        if method == 'bidirectional':
            frame_order = self.bidirectional_reconstruction(similarity_matrix)
        else:
            frame_order = self.greedy_path_reconstruction(similarity_matrix)
        
        if frame_order is None:
            print("Error: Failed to reconstruct frame order!")
            return None, 0
        
        self.save_reconstructed_video(frame_order, output_path, fps, width, height)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        log_data = {
            'execution_time_seconds': execution_time,
            'num_frames': len(self.frames),
            'method': method,
            'output_path': str(output_path),
            'success': True
        }
        
        with open('execution_log.json', 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"\n{'='*50}")
        print(f"Total execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
        print(f"{'='*50}")
        
        return frame_order, execution_time


def main():
    parser = argparse.ArgumentParser(description='Reconstruct jumbled video frames')
    parser.add_argument('input_video', help='Path to jumbled video file')
    parser.add_argument('-o', '--output', default='reconstructed_video.mp4',
                       help='Output video path (default: reconstructed_video.mp4)')
    parser.add_argument('-m', '--method', choices=['greedy', 'bidirectional'], 
                       default='greedy', help='Reconstruction method')
    
    args = parser.parse_args()
    
    reconstructor = VideoReconstructor(args.input_video)
    
    reconstructor.reconstruct(args.output, method=args.method)


if __name__ == '__main__':
    main()
