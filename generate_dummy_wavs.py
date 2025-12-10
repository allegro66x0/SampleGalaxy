import numpy as np
import soundfile as sf
import os

def generate_tone(filename, freq, duration=1.0, sr=22050):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    x = 0.5 * np.sin(2 * np.pi * freq * t)
    sf.write(filename, x, sr)

def generate_noise(filename, duration=1.0, sr=22050):
    x = 0.5 * np.random.uniform(-1, 1, int(sr * duration))
    sf.write(filename, x, sr)

def create_dummy_dataset(root_dir="dummy_samples"):
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    
    # いくつかのトーンを生成
    generate_tone(os.path.join(root_dir, "sine_440.wav"), 440)
    generate_tone(os.path.join(root_dir, "sine_880.wav"), 880)
    generate_tone(os.path.join(root_dir, "sine_220.wav"), 220)
    
    # いくつかのノイズを生成
    generate_noise(os.path.join(root_dir, "noise_1.wav"))
    generate_noise(os.path.join(root_dir, "noise_2.wav"))
    
    print(f"{root_dir} にダミーサンプルを作成しました")

if __name__ == "__main__":
    create_dummy_dataset()
