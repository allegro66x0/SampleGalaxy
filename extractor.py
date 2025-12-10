import os
import argparse
import numpy as np
import librosa
import joblib
import warnings

# 出力をきれいにするために警告を抑制
warnings.filterwarnings('ignore')

def extract_raw_features(file_path, sr=22050, duration=5.0):
    """
    WAVファイルから生の音声特徴量を抽出します。
    辞書形式で返します（スケーリングなどの後処理はしない）。
    """
    try:
        y, _ = librosa.load(file_path, sr=sr, duration=duration)
        if len(y) == 0:
            return None
        
        # 1. MFCC (音色) - 13次元
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_var = np.var(mfcc, axis=1)
        
        # 2. Spectral Centroid (音の重心/明るさ)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        centroid_mean = np.mean(centroid)
        
        # 3. Spectral Flatness (ノイズっぽさ)
        flatness = librosa.feature.spectral_flatness(y=y)
        flatness_mean = np.mean(flatness)
        
        # 4. Spectral Contrast (音のピークと谷の差) - 7次元 (6 bands + 1)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(contrast, axis=1)
        
        # 5. RMS (音量)
        rms = librosa.feature.rms(y=y)
        rms_mean = float(np.mean(rms))
        
        # 6. Chroma STFT (音階分布) - 12次元
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        
        # 7. Zero Crossing Rate (打楽器 vs 持続音)
        zcr = librosa.feature.zero_crossing_rate(y=y)
        zcr_mean = np.mean(zcr)
        
        # 8. Onset (頭出し位置) - 無音区間をスキップ
        non_silent_intervals = librosa.effects.split(y, top_db=60)
        if len(non_silent_intervals) > 0:
            start_sample = non_silent_intervals[0][0]
            start_time = float(start_sample / sr)
        else:
            start_time = 0.0

        # 9. Duration (全体の長さ)
        full_duration = librosa.get_duration(path=file_path)

        # 辞書として返す（後で重み付けしやすいように）
        return {
            "mfcc_mean": mfcc_mean,
            "mfcc_var": mfcc_var,
            "centroid_mean": centroid_mean,
            "flatness_mean": flatness_mean,
            "contrast_mean": contrast_mean,
            "rms_mean": rms_mean,
            "chroma_mean": chroma_mean,
            "zcr_mean": zcr_mean,
            "start_time": start_time,
            "duration": full_duration,
            "path": file_path
        }
    except Exception as e:
        print(f"エラー: {file_path} の処理中に問題が発生しました: {e}")
        return None

def scan_directory(root_dir):
    wav_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.wav'):
                wav_files.append(os.path.join(root, file))
    return wav_files

def main(root_dir, output_file='raw_features.pkl'):
    # 既存データのロード
    existing_data = {}
    if os.path.exists(output_file):
        print(f"既存の解析データ読み込み中: {output_file}")
        try:
            old_list = joblib.load(output_file)
            for item in old_list:
                existing_data[item['path']] = item
        except Exception as e:
            print(f"既存データの読み込み失敗: {e}")

    print(f"{root_dir} をスキャン中...")
    files = scan_directory(root_dir)
    print(f"検出されたファイル数: {len(files)}")
    
    results = []
    
    for i, f in enumerate(files):
        # 差分更新: 既存データにあればそれを使う
        if f in existing_data:
            results.append(existing_data[f])
        else:
            # 新規解析
            data = extract_raw_features(f)
            if data is not None:
                results.append(data)
                
        if (i + 1) % 100 == 0:
            print(f"処理状況: {i + 1}/{len(files)}")
            
    print(f"保存中: {output_file} (データ数: {len(results)})")
    joblib.dump(results, output_file)
    print("完了。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="特徴量抽出 (Extractor)")
    parser.add_argument("root_dir", help="WAVファイルのルートディレクトリ")
    parser.add_argument("--output", default="raw_features.pkl", help="出力ファイル (.pkl)")
    
    args = parser.parse_args()
    main(args.root_dir, args.output)
