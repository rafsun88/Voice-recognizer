import os
import numpy as np
import librosa
import soundfile as sf
from speechbrain.pretrained import SpeakerRecognition

def preprocess_audio(input_path, output_path=None, target_sr=16000, top_db=30):

    y, sr = librosa.load(input_path, sr=None)

    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # Trim leading and trailing silence
    y, _ = librosa.effects.trim(y, top_db=top_db)

    # Normalize peak volume
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak


    if output_path is None:
        filename = os.path.basename(input_path)
        output_path = os.path.join(os.path.dirname(input_path), f"cleaned_{filename}")

    # Save processed file
    sf.write(output_path, y, sr)
    print(f"✅ Processed and saved: {output_path}")
    return output_path

def verify_speakers(audio1, audio2, threshold=0.60):
    # Load pretrained model
    verification = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec"
    )

    # Process both audios
    clean1 = preprocess_audio(audio1)
    clean2 = preprocess_audio(audio2)


    score, _ = verification.verify_files(clean1, clean2)
    score_val = score.item()

    print(f"\n🔍 Similarity Score: {score_val:.4f}")
    if score_val >= threshold:
        print("🟢 Result: Same Speaker")
    else:
        print("🔴 Result: Different Speakers")


if __name__ == "__main__":
    voice1 = "owner.wav"
    voice2 = "record_out.wav"

    verify_speakers(voice1, voice2)
