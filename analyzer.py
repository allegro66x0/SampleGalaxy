import os
import json
import argparse
import numpy as np
import librosa
import soundfile as sf
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import math
import warnings

# 出力をきれいにするために警告を抑制
warnings.filterwarnings('ignore')

def remove_overlaps(coords, spread=5.0):
    """
    座標の重なりを防ぐためにグリッド状に再配置する（Spiral Search）
    coords: shape (N, 2) の numpy array
    spread: グリッドの広がり係数 (大きいほど疎になる)
    """
    n_points = len(coords)
    if n_points == 0:
        return coords

    # 1. 座標の正規化 (0.0 から grid_size)
    # グリッドサイズが指定されていない場合、点の数の平方根程度にする（正方形に近い密度）
    # spread係数を掛けて、余白を作る
    grid_size = int(math.sqrt(n_points) * spread)
    if grid_size < 100: grid_size = 100 # 最低限のサイズ
        
    # Min-Max Scaling
    min_vals = np.min(coords, axis=0)
    max_vals = np.max(coords, axis=0)
    range_vals = max_vals - min_vals
    
    # 0除算回避
    range_vals[range_vals == 0] = 1.0
    
    scaled_coords = (coords - min_vals) / range_vals # 0.0 ~ 1.0
    scaled_coords *= grid_size # 0.0 ~ grid_size
    
    # 2. グリッド配置 (Spiral Search)
    occupied = set()
    final_coords = np.zeros_like(coords)
    
    # 中心から渦巻き状に探索する関数
    def get_spiral_points(cx, cy, radius):
        # radius=0 の時は中心点のみ
        if radius == 0:
            yield (cx, cy)
            return
            
        # radius > 0 の時、周囲の正方形の辺を走査
        # 上辺
        for x in range(cx - radius, cx + radius + 1):
            yield (x, cy - radius)
        # 右辺
        for y in range(cy - radius + 1, cy + radius + 1):
            yield (cx + radius, y)
        # 下辺
        for x in range(cx + radius - 1, cx - radius - 1, -1):
            yield (x, cy + radius)
        # 左辺
        for y in range(cy + radius - 1, cy - radius, -1):
            yield (cx - radius, y)

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
            # 無限ループ防止（念のため）
            if radius > grid_size * 2:
                # print(f"各点配置警告: {i}番目の点を配置できませんでした。")
                final_coords[i] = [ix, iy] # 諦めて重ねる
                break
                
    return final_coords

def extract_features(file_path, sr=22050, duration=5.0):
    """
    WAVファイルから音声特徴量を抽出します。
    """
    try:
        y, _ = librosa.load(file_path, sr=sr, duration=duration)
        if len(y) == 0:
            return None
        
        # 特徴量抽出
        # 1. MFCC (音色) - 元に戻す: 20 -> 13次元
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_var = np.var(mfcc, axis=1)
        
        # 2. Spectral Centroid (音の重心/明るさ)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        centroid_mean = np.mean(centroid)
        
        # 3. Spectral Flatness (ノイズっぽさ)
        flatness = librosa.feature.spectral_flatness(y=y)
        flatness_mean = np.mean(flatness)
        
        # 4. Spectral Contrast (音のピークと谷の差)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(contrast, axis=1)
        
        # --- 新機能: 音量と頭出し ---
        # 5. RMS (音量)
        rms = librosa.feature.rms(y=y)
        rms_mean = float(np.mean(rms))
        
        # --- 追加機能: Melodic & Loop Separation ---
        # 6. Chroma STFT (音階分布 - 12次元)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        
        # 7. Zero Crossing Rate (打楽器 vs 持続音)
        zcr = librosa.feature.zero_crossing_rate(y=y)
        zcr_mean = np.mean(zcr)
        
        # 6. Onset (頭出し位置) - 無音区間をスキップ
        # top_db=60 で無音判定
        non_silent_intervals = librosa.effects.split(y, top_db=60)
        if len(non_silent_intervals) > 0:
            start_sample = non_silent_intervals[0][0]
            start_time = float(start_sample / sr)
        else:
            start_time = 0.0

        # 7. Duration (全体の長さ)
        # librosa.loadでduration指定していると切り取られるので、ファイルから直接取得
        full_duration = librosa.get_duration(path=file_path)

        # 全特徴量を結合
        features = np.concatenate([
            mfcc_mean, 
            mfcc_var, 
            [centroid_mean], 
            [flatness_mean], 
            contrast_mean,
            chroma_mean,
            [zcr_mean]
        ])
        
        return {
            "features": features,
            "rms": rms_mean,
            "start_time": start_time,
            "duration": full_duration
        }
    except Exception as e:
        print(f"エラー: {file_path} の処理中に問題が発生しました: {e}")
        return None

def classify_sample(file_path, duration, features):
    """
    ファイル名、特徴量、長さからサンプルを詳細に分類します。
    """
    name = os.path.basename(file_path).lower()
    
    # 0. BPM / Loop 判定
    # "bpm" が含まれる、または長尺 (ユーザー定義: > 4.0s)
    # 数字+bpm のパターン (e.g. 120bpm) も考慮したいが単純な "bpm" 検索でカバー
    if "bpm" in name or "loop" in name or duration > 4.0:
        if "top" in name:
            # トップループはドラムループの一種として扱うか、Loopとして扱うか
            return "LOOP" # ID: 9
        return "LOOP" # ID: 9

    # 1. 楽器判定 (Melodic)
    if any(x in name for x in ["guitar", "gtr", "acoustic", "electric"]):
        return "GUITAR" # ID: 7
        
    if any(x in name for x in ["piano", "key", "synth", "rhodes", "organ"]):
        return "PIANO" # ID: 8
        
    if "bass" in name or "808" in name:
        return "BASS" # ID: 6

    # 2. ドラム/Percussion判定
    if "kick" in name or "bd" in name:
        if duration > 0.8: return "BASS" # Long Kick -> Bass
        return "KICK" # ID: 0
        
    if any(x in name for x in ["snare", "sd"]):
        return "SNARE" # ID: 1
        
    if any(x in name for x in ["clap", "cp"]):
        return "CLAP" # ID: 2
        
    if any(x in name for x in ["hat", "hh", "oh", "ch"]):
        return "HIHAT" # ID: 3
        
    if any(x in name for x in ["crash", "cymbal", "ride", "cy"]):
        return "CRASH" # ID: 4
        
    if any(x in name for x in ["tom", "tm"]):
        return "TOM" # ID: 5
        
    if "perc" in name:
        return "FX" 
        
    if any(x in name for x in ["fx", "sfx", "riser", "downer", "vox", "vocal"]):
        return "FX" # ID: 10
        
    return "UNKNOWN" # ID: -1 (Before was OTHER)

def scan_directory(root_dir):
    """
    ディレクトリを再帰的にスキャンして .wav ファイルを探します。
    """
    wav_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.wav'):
                wav_files.append(os.path.join(root, file))
    return wav_files

def analyze_samples(root_dir, output_file='database.json', spread=5.0):
    # A. 既存データのロード
    existing_data = {}
    if os.path.exists(output_file):
        print(f"既存のデータベースを読み込んでいます: {output_file}")
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # パスをキーにした辞書に変換
                for item in data:
                    existing_data[item['path']] = item
        except Exception as e:
            print(f"データベースの読み込みに失敗しました (新規作成します): {e}")

    print(f"{root_dir} をスキャン中...")
    new_files = scan_directory(root_dir)
    print(f"{len(new_files)} 個の新規スキャンファイルが見つかりました。")
    
    # B. スキャン結果と既存データの統合
    # パスで一意にする
    # 既存データにあるファイルは維持し、今回スキャンされたファイルを追加/更新する
    
    all_paths = set(existing_data.keys()).union(set(new_files))
    all_paths = sorted(list(all_paths)) # 再現性のためソート
    
    print(f"総ファイル数: {len(all_paths)}")
    
    feature_list = []
    metadata_list = []
    valid_files = [] 
    
    # C. 特徴量抽出 (またはキャッシュ利用)
    print("特徴量を準備中...")
    for i, f in enumerate(all_paths):
        # 今回スキャンされたファイルなら強制再解析 (Update)
        is_new_scan = (f in new_files)
        
        # キャッシュが使える条件:
        # - 既存データにある
        # - 'features' キーが存在する
        # - 今回スキャンされたファイルではない
        
        if not is_new_scan and (f in existing_data) and ("features" in existing_data[f]):
            # キャッシュ利用
            item = existing_data[f]
            try:
                # featuresがリスト形式で保存されているはず
                feat = np.array(item["features"])
                
                # メタデータ復元
                feature_list.append(feat)
                metadata_list.append({
                    "rms": item.get("rms", 0),
                    "start_time": item.get("start_time", 0),
                    "duration": item.get("duration", 0),
                    "category": item.get("category", "UNKNOWN")
                })
                valid_files.append(f)
            except Exception as e:
                 print(f"キャッシュ読み込みエラー (再解析します): {f} - {e}")
                 # フォールバック
                 is_new_scan = True 

        if is_new_scan or (f not in existing_data) or ("features" not in existing_data[f]):
            # 新規解析 or 更新 or キャッシュなし
            if not os.path.exists(f):
                continue
                
            result = extract_features(f)
            if result is not None:
                category = classify_sample(f, result["duration"], result["features"])
                
                feature_list.append(result["features"])
                metadata_list.append({
                    "rms": result["rms"],
                    "start_time": result["start_time"],
                    "duration": result["duration"],
                    "category": category
                })
                valid_files.append(f)
                
        if (i + 1) % 100 == 0:
             print(f"処理状況: {i + 1}/{len(all_paths)}")

    if not feature_list:
        print("有効なデータがありませんでした。")
        return

    X = np.array(feature_list)
    
    # t-SNEのために十分なサンプルがあるか確認
    n_samples = X.shape[0]
    
    # t-SNE parameters
    n_components = 2
    perplexity = min(30, n_samples - 1) # データ数が少ない場合は下げる
    if perplexity < 5: perplexity = max(1, n_samples - 1)
    
    if n_samples < 2:
         print("サンプル不足で次元削減をスキップします。")
         embedding = np.random.rand(n_samples, 2)
    else:
        print("t-SNEで次元削減中...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # t-SNE実行
        # metric='cosine' が音声特徴量に効くことが多い
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            init='pca', # PCA初期化で安定させる
            random_state=42,
            metric='cosine', # または 'euclidean'
        )
        embedding = tsne.fit_transform(X_scaled)

    # 重なり防止 (Grid Snapping & Spiral Search)
    print("重なりを除去してグリッドに配置中...")
    embedding = remove_overlaps(embedding, spread=spread)
    
    print("K-Meansでクラスタリング中...")
    n_clusters = min(8, n_samples)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embedding)
    
    # データを保存 (特徴量も含める)
    output_data = []
    for i in range(n_samples):
        meta = metadata_list[i]
        output_data.append({
            "path": valid_files[i],
            "x": float(embedding[i, 0]),
            "y": float(embedding[i, 1]),
            "cluster": int(clusters[i]),
            "rms": meta["rms"],
            "start_time": meta["start_time"],
            "duration": meta["duration"],
            "category": meta["category"],
            "features": feature_list[i].tolist() # 次回のためにキャッシュ
        })
        
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
        
    print(f"分析完了。 {output_file} を更新しました (登録数: {len(output_data)})。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="音声サンプルアナライザー")
    parser.add_argument("root_dir", help="WAVファイルをスキャンするルートディレクトリ")
    parser.add_argument("--output", default="database.json", help="出力JSONファイル")
    parser.add_argument("--spread", type=float, default=5.0, help="分布の広がり具合 (デフォルト: 5.0。大きいほど疎)")
    
    args = parser.parse_args()
    analyze_samples(args.root_dir, args.output, args.spread)
