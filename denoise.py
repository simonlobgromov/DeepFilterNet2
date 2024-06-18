import tempfile
import torch
from torch import Tensor
from df import config
from df.enhance import enhance, init_df, load_audio, save_audio
from df.io import resample


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, df, _ = init_df("./DeepFilterNet2", config_allow_defaults=True)
model = model.to(device=device).eval()

def demo_fn(file_path: str, snr: int=10):
    sr = config("sr", 48000, int, section="df")
    snr = int(snr)
    max_s = 10  # limit to 10 seconds
    
    # Load the audio file
    sample, meta = load_audio(file_path, sr)
    max_len = max_s * sr
    if sample.shape[-1] > max_len:
        start = torch.randint(0, sample.shape[-1] - max_len, ()).item()
        sample = sample[..., start : start + max_len]
    
    if sample.dim() > 1 and sample.shape[0] > 1:
        assert sample.shape[1] > sample.shape[0], f"Expecting channels first, but got {sample.shape}"
        sample = sample.mean(dim=0, keepdim=True)
    
    # Enhance the audio
    enhanced = enhance(model, df, sample)
    
    lim = torch.linspace(0.0, 1.0, int(sr * 0.15)).unsqueeze(0)
    lim = torch.cat((lim, torch.ones(1, enhanced.shape[1] - lim.shape[1])), dim=1)
    enhanced = enhanced * lim
    
    if meta.sample_rate != sr:
        enhanced = resample(enhanced, sr, meta.sample_rate)
        sample = resample(sample, sr, meta.sample_rate)
        sr = meta.sample_rate
    
    # Save the noisy and enhanced audio files
    noisy_wav = tempfile.NamedTemporaryFile(suffix="noisy.wav", delete=False).name
    save_audio(noisy_wav, sample, sr)
    enhanced_wav = tempfile.NamedTemporaryFile(suffix="enhanced.wav", delete=False).name
    save_audio(enhanced_wav, enhanced, sr)
    

    
    return noisy_wav, enhanced_wav
