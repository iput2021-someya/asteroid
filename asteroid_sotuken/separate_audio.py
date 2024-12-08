import torchaudio
from asteroid.models import BaseModel
import torch

def separate_audio(input_path, model_name, output_dir):
    # モデルのロード
    model = BaseModel.from_pretrained(model_name)
    print(f"Loaded model: {model_name}")

    # 音声ファイルを読み込む
    waveform, sample_rate = torchaudio.load(input_path)
    print(f"Loaded audio file: {input_path}, Sample rate: {sample_rate}")

    # 音声分離
    with torch.no_grad():
        separated = model.separate(waveform)

    # 出力を保存
    for i, src in enumerate(separated):
        output_path = f"{output_dir}/source_{i}.wav"
        torchaudio.save(output_path, src.unsqueeze(0), sample_rate)
        print(f"Source {i} saved to {output_path}")

# パラメータ設定
input_audio_path = "path/to/your/mixed_audio.wav"
pretrained_model_name = "mpariente/ConvTasNet_WHAM_sepclean"
output_directory = "output"

separate_audio(input_audio_path, pretrained_model_name, output_directory)
