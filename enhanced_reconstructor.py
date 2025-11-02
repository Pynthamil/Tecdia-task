import cv2
import numpy as np
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor
import json
from tqdm import tqdm
import argparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

class EnhancedVideoReconstructor:
    def __init__(self, video_path, use_multiprocessing=True, num_workers=None):
        self.video_path = Path(video_path)
        self.use_multiprocessing = use_multiprocessing
        self.num_workers = num_workers
        self.frames = []
        self.frame_features = []
        
    def load_frames(self):
        print("Loading frames and extracting features...")
        cap = cv2.VideoCapture(str(self.video_path))
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video: {frame_count} frames, {fps} FPS, {width}x{height}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            resized = cv2.resize(frame, (320, 180))
            self.frames.append(resized)
            
            features = self.extract_features(resized)
            self.frame_features.append(features)
        
        cap.release()
        print(f"Loaded {len(self.frames)} frames")
        return fps, width, height
    
    def extract_features(self, frame):
        hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], 
                           [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges) / edges.size
        
        brightness = np.mean(gray)
        
        features = np.concatenate([hist, [edge_density, brightness]])
        return features
    
    def cluster_frames(self, n_clusters=10):
        print(f"\nClustering frames into {n_clusters} groups...")
        
        feature_matrix = np.array(self.frame_features)
        
        pca = PCA(n_components=min(50, feature_matrix.shape[1]))
        reduced_features = pca.fit_transform(feature_matrix)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(reduced_features)
        
        clusters = {}
        for idx, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(idx)
        
        print(f"Created {len(clusters)} clusters")
        for label, frames in clusters.items():
            print(f"  Cluster {label}: {len(frames)} frames")
        
        return clusters, cluster_labels
    
    def compute_frame_similarity_fast(self, idx1, idx2):
        feat1 = self.frame_features[idx1]
        feat2 = self.frame_features[idx2]
        
        similarity = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
        
        return similarity
    
    def compute_frame_similarity_accurate(self, idx1, idx2):
        frame1 = self.frames[idx1]
        frame2 = self.frames[idx2]
        
        hist1 = cv2.calcHist([frame1], [0, 1, 2], None, [8, 8, 8], 
                            [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([frame2], [0, 1, 2], None, [8, 8, 8], 
                            [0, 256, 0, 256, 0, 256])
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()
        hist_sim = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        mse = np.mean((frame1.astype(float) - frame2.astype(float)) ** 2)
        mse_sim = 1.0 / (1.0 + mse / 1000.0)
        
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        mu1 = np.mean(gray1)
        mu2 = np.mean(gray2)
        
        var1 = np.var(gray1)
        var2 = np.var(gray2)
        cov = np.mean((gray1 - mu1) * (gray2 - mu2))
        
        c1, c2 = 6.5025, 58.5225
        ssim = ((2 * mu1 * mu2 + c1) * (2 * cov + c2)) / \
               ((mu1**2 + mu2**2 + c1) * (var1 + var2 + c2))
        ssim = (ssim + 1) / 2
        
        return 0.4 * hist_sim + 0.3 * mse_sim + 0.3 * ssim
    
    def build_similarity_matrix_adaptive(self):
        n = len(self.frames)
        print(f"\nBuilding adaptive similarity matrix for {n} frames...")
        
        similarity_matrix = np.zeros((n, n))
        
        print("Phase 1: Computing fast feature similarities...")
        for i in tqdm(range(n)):
            for j in range(i + 1, n):
                sim = self.compute_frame_similarity_fast(i, j)
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
        
        np.fill_diagonal(similarity_matrix, 1.0)
        
        print("Phase 2: Refining top similarities...")
        k = 20
        
        for i in tqdm(range(n)):
            top_k_indices = np.argsort(similarity_matrix[i])[-k-1:-1]
            
            for j in top_k_indices:
                if i < j:
                    accurate_sim = self.compute_frame_similarity_accurate(i, j)
                    similarity_matrix[i, j] = accurate_sim
                    similarity_matrix[j, i] = accurate_sim
        
        return similarity_matrix
    
    def greedy_reconstruction_with_lookahead(self, similarity_matrix, lookahead=3):
        print(f"\nReconstructing with lookahead={lookahead}...")
        n = len(similarity_matrix)
        
        best_path = None
        best_score = -np.inf
        
        num_trials = min(10, n)
        start_candidates = np.linspace(0, n-1, num_trials, dtype=int)
        
        for start_idx in tqdm(start_candidates, desc="Trying starts"):
            path = [start_idx]
            visited = {start_idx}
            
            while len(visited) < n:
                current = path[-1]
                
                similarities = similarity_matrix[current].copy()
                similarities[list(visited)] = -np.inf
                
                top_candidates = np.argsort(similarities)[-lookahead:]
                
                best_candidate = None
                best_candidate_score = -np.inf
                
                for candidate in top_candidates:
                    if candidate in visited:
                        continue
                    
                    next_similarities = similarity_matrix[candidate].copy()
                    next_similarities[list(visited | {candidate})] = -np.inf
                    
                    if len(visited) + 1 < n:
                        lookahead_score = (similarities[candidate] + 
                                         np.max(next_similarities))
                    else:
                        lookahead_score = similarities[candidate]
                    
                    if lookahead_score > best_candidate_score:
                        best_candidate_score = lookahead_score
                        best_candidate = candidate
                
                if best_candidate is not None:
                    path.append(best_candidate)
                    visited.add(best_candidate)
                else:
                    break
            
            if len(path) == n:
                score = sum(similarity_matrix[path[i], path[i+1]] 
                           for i in range(len(path)-1))
                if score > best_score:
                    best_score = score
                    best_path = path
        
        print(f"Best path score: {best_score:.4f}")
        return best_path
    
    def two_opt_refinement(self, path, similarity_matrix, max_iterations=100):
        print("\nRefining path with 2-opt...")
        n = len(path)
        improved = True
        iteration = 0
        
        def path_score(p):
            return sum(similarity_matrix[p[i], p[i+1]] for i in range(len(p)-1))
        
        best_score = path_score(path)
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            for i in range(1, n - 2):
                for j in range(i + 1, n - 1):
                    new_path = path[:i] + path[i:j+1][::-1] + path[j+1:]
                    new_score = path_score(new_path)
                    
                    if new_score > best_score:
                        path = new_path
                        best_score = new_score
                        improved = True
                        break
                
                if improved:
                    break
        
        print(f"Refinement improved score to: {best_score:.4f}")
        return path
    
    def save_reconstructed_video(self, frame_order, output_path, 
                                original_fps, original_width, original_height):
        print(f"\nSaving video to {output_path}...")
        
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
        
        for idx in tqdm(frame_order, desc="Writing"):
            out.write(original_frames[idx])
        
        out.release()
        print("Video saved!")
    
    def reconstruct(self, output_path='reconstructed_video.mp4', 
                   use_refinement=True):
        start_time = time.time()
        
        fps, width, height = self.load_frames()
        
        similarity_matrix = self.build_similarity_matrix_adaptive()
        
        frame_order = self.greedy_reconstruction_with_lookahead(similarity_matrix)
        
        if use_refinement and frame_order is not None:
            frame_order = self.two_opt_refinement(frame_order, similarity_matrix)
        
        if frame_order:
            self.save_reconstructed_video(frame_order, output_path, 
                                        fps, width, height)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        log_data = {
            'execution_time_seconds': execution_time,
            'num_frames': len(self.frames),
            'refinement': use_refinement,
            'output_path': str(output_path)
        }
        
        with open('execution_log.json', 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"\n{'='*50}")
        print(f"Total time: {execution_time:.2f}s ({execution_time/60:.2f}min)")
        print(f"{'='*50}")
        
        return frame_order


def main():
    parser = argparse.ArgumentParser(description='Enhanced video reconstruction')
    parser.add_argument('input_video', help='Jumbled video file')
    parser.add_argument('-o', '--output', default='reconstructed_video.mp4',
                       help='Output path')
    parser.add_argument('--no-refinement', action='store_true',
                       help='Skip 2-opt refinement')
    
    args = parser.parse_args()
    
    reconstructor = EnhancedVideoReconstructor(args.input_video)
    reconstructor.reconstruct(args.output, use_refinement=not args.no_refinement)


if __name__ == '__main__':
    main()
