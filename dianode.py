import torchaudio
import numpy as np
import torch
# import comfy.model_management
import folder_paths
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from dia.model import Dia


models_dir = folder_paths.models_dir
model_path = os.path.join(models_dir, "TTS")
config_path = os.path.join(model_path, "Dia-1.6B", "config.json")
checkpoint_path = os.path.join(model_path, "Dia-1.6B", "dia-v0_1.pth")
dac_model_path = os.path.join(model_path, "DAC.speech.v1.0", "weights_44khz_8kbps_0.0.1.pth")

class DiaTTSRun:
    def __init__(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        self.model_cache = None
        self.dac_model = None
        self.device = torch.device(device)
        print(f"Using device: {device}")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text_input": ("STRING", {"forceInput": True}),
                "max_new_tokens": ("INT", {"default": 2000, "min": 860, "max": 3072, "step": 50}),
                "cfg_scale": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 5.0, "step": 0.1}),
                "temperature": ("FLOAT", {"default": 1.3, "min": 1.0, "max": 1.5, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.80, "max": 1.0, "step": 0.01}),
                "cfg_filter_top_k": ("INT", {"default": 30, "min": 15, "max": 50, "step": 1}),
                "speed_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "unload_model": ("BOOLEAN", {"default": True}),
                # "use_torch_compile": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "audio_prompt_input": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "run_inference"
    CATEGORY = "🎤MW/MW-Dia"

    def run_inference(
        self,
        text_input: str,
        max_new_tokens: int,
        cfg_scale: float,
        temperature: float,
        top_p: float,
        cfg_filter_top_k: int,
        speed_factor: float,
        unload_model: bool,
        use_torch_compile=False,
        audio_prompt_input=None,
    ):
        if self.model_cache is None or self.dac_model is None:
            import dac
            self.dac_model = dac.DAC.load(dac_model_path).to(self.device).eval()
            self.model_cache = Dia.from_local(config_path, checkpoint_path, self.dac_model, self.device)
        if audio_prompt_input is None:
            audio_sr = None
        else:
            audio_data = audio_prompt_input["waveform"].squeeze(0)
            sr = audio_prompt_input["sample_rate"]

            audio_sr = (audio_data, sr)
        # Use torch.inference_mode() context manager for the generation call
        with torch.inference_mode():
            output_audio_np = self.model_cache.generate(
                text_input,
                max_tokens=max_new_tokens,
                cfg_scale=cfg_scale,
                temperature=temperature,
                top_p=top_p,
                use_cfg_filter=True,
                cfg_filter_top_k=cfg_filter_top_k,
                use_torch_compile=use_torch_compile, 
                audio_sr=audio_sr,
            )

        if output_audio_np is not None:
            # Get sample rate from the loaded DAC model
            output_sr = 44100

            # --- Slow down audio ---
            original_len = len(output_audio_np)
            # Ensure speed_factor is positive and not excessively small/large to avoid issues
            speed_factor = max(0.1, min(speed_factor, 5.0))
            target_len = int(
                original_len / speed_factor
            )  # Target length based on speed_factor
            if (
                target_len != original_len and target_len > 0
            ):  # Only interpolate if length changes and is valid
                x_original = np.arange(original_len)
                x_resampled = np.linspace(0, original_len - 1, target_len)
                resampled_audio_np = np.interp(x_resampled, x_original, output_audio_np)
                output_audio = resampled_audio_np.astype(np.float32)
                print(
                    f"Resampled audio from {original_len} to {target_len} samples for {speed_factor:.2f}x speed."
                )
            else:
                output_audio =  output_audio_np # Keep original if calculation fails or no change
                print(f"Skipping audio speed adjustment (factor: {speed_factor:.2f}).")
            # --- End slowdown ---

            print(
                f"Audio conversion successful. Final shape: {output_audio.shape}, Sample Rate: {output_sr}"
            )
            # Unload model if requested
            if unload_model:
                self.model_cache = None
                self.dac_model = None
                torch.cuda.empty_cache()

            return ({"waveform": torch.from_numpy(output_audio).unsqueeze(0).unsqueeze(0), "sample_rate": output_sr},)

        else:
            raise  RuntimeError("Audio generation failed.")



NODE_CLASS_MAPPINGS = {
    "DiaTTSRun": DiaTTSRun,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DiaTTSRun": "DiaTTS Run",
}