import os
import argparse
import json
import joblib
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import warnings

warnings.filterwarnings('ignore')

def classify_sample(file_path, duration):
    """
    ファイル名と長さからサンプルを詳細に分類します。
    """
    name = os.path.basename(file_path).lower()
    
    # 0. BPM / Loop 判定
    if "bpm" in name or "loop" in name or duration > 4.0:
        if "top" in name: return "LOOP"
        return "LOOP"

    # 1. 楽器判定 (Melodic)
    if any(x in name for x in ["guitar", "gtr", "acoustic", "electric"]): return "GUITAR"
    if any(x in name for x in ["piano", "key", "synth", "rhodes", "organ"]): return "PIANO"
    if "bass" in name or "808" in name: return "BASS"

    # 2. ドラム/Percussion判定
    if "kick" in name or "bd" in name:
        if duration > 0.8: return "BASS"
        return "KICK"
    if any(x in name for x in ["snare", "sd"]): return "SNARE"
    if any(x in name for x in ["clap", "cp"]): return "CLAP"
    if any(x in name for x in ["hat", "hh", "oh", "ch"]): return "HIHAT"
    if any(x in name for x in ["crash", "cymbal", "ride", "cy"]): return "CRASH"
    if any(x in name for x in ["tom", "tm"]): return "TOM"
    if "perc" in name: return "FX" 
    if any(x in name for x in ["fx", "sfx", "riser", "downer", "vox", "vocal"]): return "FX"
        
    return "UNKNOWN"

def remove_overlaps(coords, spread=5.0):
    n_points = len(coords)
    if n_points == 0: return coords

    grid_size = int(math.sqrt(n_points) * spread)
    if grid_size < 100: grid_size = 100
        
    # Min-Max Scaling
    min_vals = np.min(coords, axis=0)
    max_vals = np.max(coords, axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1.0
    
    scaled_coords = (coords - min_vals) / range_vals 
    scaled_coords *= grid_size
    
    occupied = set()
    final_coords = np.zeros_like(coords)
    
    def get_spiral_points(cx, cy, radius):
        if radius == 0:
            yield (cx, cy)
            return
        for x in range(cx - radius, cx + radius + 1): yield (x, cy - radius)
        for y in range(cy - radius + 1, cy + radius + 1): yield (cx + radius, y)
        for x in range(cx + radius - 1, cx - radius - 1, -1): yield (x, cy + radius)
        for y in range(cy + radius - 1, cy - radius, -1): yield (cx - radius, y)

    print(f"重なり除去処理中 (Grid: {grid_size}x{grid_size}, Spread: {spread})...")
    
    for i in range(n_points):
        x, y = scaled_coords[i]
        ix, iy = int(round(x)), int(round(y))
        
        radius = 0
        found = False
        while not found:
            for tx, ty in get_spiral_points(ix, iy, radius):
                if (tx, ty) not in occupied:
                    occupied.add((tx, ty))
                    final_coords[i] = [tx, ty]
                    found = True
                    break
            radius += 1
            if radius > grid_size * 2:
                final_coords[i] = [ix, iy]
                break
                
    return final_coords

def main(input_file='raw_features.pkl', output_file='database.json', spread=5.0):
    if not os.path.exists(input_file):
        print(f"入力ファイルが見つかりません: {input_file}")
        print("先に extractor.py を実行してください。")
        return

    print("特徴量データをロード中...")
    raw_data = joblib.load(input_file)
    n_samples = len(raw_data)
    print(f"データ数: {n_samples}")
    
    if n_samples == 0:
        return

    # --- 特徴量ベクトルの構築 (Weighting) ---
    feature_matrix = []
    
    # 重み設定 (実験用パラメータ)
    W_MFCC = 1.0
    W_CHROMA = 2.0  # 音階情報を重視するか？
    W_CONTRAST = 1.0
    W_FLATNESS = 1.0
    W_ZCR = 1.0
    
    for item in raw_data:
        # ベクトル結合
        vec = np.concatenate([
            item["mfcc_mean"] * W_MFCC,
            item["mfcc_var"] * W_MFCC * 0.5, # 分散は少し弱める
            [item["centroid_mean"]],
            [item["flatness_mean"] * W_FLATNESS],
            item["contrast_mean"] * W_CONTRAST,
            item["chroma_mean"] * W_CHROMA,
            [item["zcr_mean"] * W_ZCR]
        ])
        feature_matrix.append(vec)

    X = np.array(feature_matrix)
    
    # --- t-SNE ---
    n_components = 2
    perplexity = min(30, n_samples - 1)
    if perplexity < 5: perplexity = max(1, n_samples - 1)

    print("t-SNEで次元削減中...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        init='pca',
        random_state=42,
        metric='cosine',
        n_jobs=-1 # 可能なら並列化
    )
    embedding = tsne.fit_transform(X_scaled)
    
    # --- Grid Layout ---
    embedding = remove_overlaps(embedding, spread=spread)
    
    # --- Clustering ---
    print("K-Meansでクラスタリング中...")
    n_clusters = min(8, n_samples)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embedding)
    
    # --- Export ---
    output_data = []
    for i in range(n_samples):
        item = raw_data[i]
        category = classify_sample(item["path"], item["duration"])
        
        output_data.append({
            "path": item["path"],
            "x": float(embedding[i, 0]),
            "y": float(embedding[i, 1]),
            "cluster": int(clusters[i]),
            "rms": item["rms_mean"],
            "start_time": item["start_time"],
            "duration": item["duration"],
            "category": category,
            "centroid": item["centroid_mean"] 
            # GUI側で特徴量を使わないなら保存不要、使うなら raw_features か X を保存
        })
        
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
        
    print(f"完了: {output_file} (登録数: {len(output_data)})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ソーター (Sorter)")
    parser.add_argument("--input", default="raw_features.pkl", help="入力ファイル (.pkl)")
    parser.add_argument("--output", default="database.json", help="出力ファイル (.json)")
    parser.add_argument("--spread", type=float, default=5.0, help="分布の広がり (デフォルト: 5.0)")
    
    args = parser.parse_args()
    main(args.input, args.output, args.spread)
