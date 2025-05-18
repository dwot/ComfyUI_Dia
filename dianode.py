import numpy as np
import torch
import librosa
# import comfy.model_management
import folder_paths
import os
import sys
import tempfile
import torchaudio
from typing import Optional, List, Union
import re

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from dia.model import Dia


models_dir = folder_paths.models_dir
model_path = os.path.join(models_dir, "TTS")
config_path = os.path.join(model_path, "Dia-1.6B", "config.json")
checkpoint_path = os.path.join(model_path, "Dia-1.6B")
dac_model_path = os.path.join(model_path, "DAC.speech.v1.0", "weights_44khz_8kbps_0.0.1.pth")
cache_dir = folder_paths.get_temp_directory()

def cache_audio_tensor(
    cache_dir,
    audio_tensor: torch.Tensor,
    sample_rate: int,
    filename_prefix: str = "cached_audio_",
    audio_format: Optional[str] = ".wav"
) -> str:
    try:
        with tempfile.NamedTemporaryFile(
            prefix=filename_prefix,
            suffix=audio_format,
            dir=cache_dir,
            delete=False 
        ) as tmp_file:
            temp_filepath = tmp_file.name
        
        torchaudio.save(temp_filepath, audio_tensor, sample_rate)

        return temp_filepath
    except Exception as e:
        raise Exception(f"Error caching audio tensor: {e}")


class DiaTTSRun:
    def __init__(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        self.model_cache = None
        self.device = torch.device(device)
        print(f"Using device: {device}")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text_input": ("STRING", {"forceInput": True}),
                "max_new_tokens": ("INT", {"default": 3000, "min": 860, "max": 3072, "step": 2}),
                "cfg_scale": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 5.0, "step": 0.1}),
                "temperature": ("FLOAT", {"default": 1.3, "min": 1.0, "max": 1.5, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.80, "max": 1.0, "step": 0.01}),
                "cfg_filter_top_k": ("INT", {"default": 30, "min": 15, "max": 50, "step": 1}),
                # "speed_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),
                # "use_torch_compile": ("BOOLEAN", {"default": True}),
                "unload_model": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "speakers_audio_input": ("AUDIO",),
                "speakers_text_input": ("STRING", {"multiline": True, "default": ""}),
                "save_speakers": ("BOOLEAN", {"default": True}),
                "speaker_id": ("STRING", {"default": "A_and_B"}),
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
        # speed_factor: float,
        unload_model: bool,
        use_torch_compile=False,
        speakers_audio_input=None,
        speakers_text_input="",
        save_speakers=False,
        speaker_id="A_and_B",
    ):
        if self.model_cache is None:
            model_path = os.path.join(checkpoint_path, "dia-v0_1.pth")
            self.model_cache = Dia.from_local(config_path, model_path, compute_dtype="float16", device=self.device, dac_model_path=dac_model_path)

        text = re.split(r'\n\s*\n', text_input.strip())
        # print(f"Input text: {text}")
        if speakers_audio_input is None:
            audio_prompts = None
        else:
            if speakers_text_input.strip() == "":
                raise ValueError("Text clone input is empty.")
            audio_data = speakers_audio_input["waveform"].squeeze(0)
            sr = speakers_audio_input["sample_rate"]

            speakers_path = os.path.join(checkpoint_path, "speakers")
            if not os.path.exists(speakers_path):
                os.makedirs(speakers_path)

            if save_speakers:
                audio_path = os.path.join(speakers_path, f"{speaker_id}.wav")
                text_path = os.path.join(speakers_path, f"{speaker_id}.txt")
                torchaudio.save(audio_path, audio_data, sr)
                with open(text_path, "w", encoding="utf-8") as f:
                    f.write(speakers_text_input.strip())
            else:
                audio_path = cache_audio_tensor(
                    cache_dir,
                    audio_data,
                    sr,
                )

            audio_prompt = self.model_cache.load_audio(audio_path)
            text = [speakers_text_input.strip() + f"\n{i}" for i in text]
            audio_prompts = [audio_prompt for i in range(len(text))]

        # Use torch.inference_mode() context manager for the generation call
        with torch.inference_mode():
            output_audio_np = self.model_cache.generate(
                text=text,
                max_tokens=max_new_tokens,
                cfg_scale=cfg_scale,
                temperature=temperature,
                top_p=top_p,
                cfg_filter_top_k=cfg_filter_top_k,
                use_torch_compile=use_torch_compile, 
                audio_prompt=audio_prompts,
                verbose=True,
            )

        if output_audio_np[0] is not None:
            # Get sample rate from the loaded DAC model
            output_sr = 44100

            output_audio = np.concatenate(output_audio_np).astype(np.float32)
            output_audio = np.clip(output_audio, -1.0, 1.0)

            # Unload model if requested
            if unload_model:
                self.model_cache = None
                torch.cuda.empty_cache()

            return ({"waveform": torch.from_numpy(output_audio).unsqueeze(0).unsqueeze(0), "sample_rate": output_sr},)

        else:
            raise  RuntimeError("Audio generation failed.")


def get_all_files(
    root_dir: str,
    return_type: str = "list",
    extensions: Optional[List[str]] = None,
    exclude_dirs: Optional[List[str]] = None,
    relative_path: bool = False
) -> Union[List[str], dict]:
    """
    递归获取目录下所有文件路径
    
    :param root_dir: 要遍历的根目录
    :param return_type: 返回类型 - "list"(列表) 或 "dict"(按目录分组)
    :param extensions: 可选的文件扩展名过滤列表 (如 ['.py', '.txt'])
    :param exclude_dirs: 要排除的目录名列表 (如 ['__pycache__', '.git'])
    :param relative_path: 是否返回相对路径 (相对于root_dir)
    :return: 文件路径列表或字典
    """
    file_paths = []
    file_dict = {}
    
    # 规范化目录路径
    root_dir = os.path.normpath(root_dir)
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 处理排除目录
        if exclude_dirs:
            dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        
        current_files = []
        for filename in filenames:
            # 扩展名过滤
            if extensions:
                if not any(filename.lower().endswith(ext.lower()) for ext in extensions):
                    continue
            
            # 构建完整路径
            full_path = os.path.join(dirpath, filename)
            
            # 处理相对路径
            if relative_path:
                full_path = os.path.relpath(full_path, root_dir)
            
            current_files.append(full_path)
        
        if return_type == "dict":
            # 使用相对路径或绝对路径作为键
            dict_key = os.path.relpath(dirpath, root_dir) if relative_path else dirpath
            if current_files:
                file_dict[dict_key] = current_files
        else:
            file_paths.extend(current_files)
    
    return file_dict if return_type == "dict" else file_paths

def get_speakers():
    speakers_dir = os.path.join(checkpoint_path, "speakers")
    if not os.path.exists(speakers_dir):
        os.makedirs(speakers_dir, exist_ok=True)
        return []
    speakers = get_all_files(speakers_dir, extensions=[".wav"], relative_path=True)
    return speakers

class DiaSpeakersPreview:
    def __init__(self):
        self.speakers_dir = os.path.join(checkpoint_path, "speakers")
    @classmethod
    def INPUT_TYPES(s):
        speakers = get_speakers()
        return {
            "required": {"speaker":(speakers,),},}

    RETURN_TYPES = ("AUDIO", "STRING",)
    RETURN_NAMES = ("audio", "text",)
    FUNCTION = "preview"
    CATEGORY = "🎤MW/MW-Dia"

    def preview(self, speaker):
        wav_path = os.path.join(self.speakers_dir, speaker)
        waveform, sample_rate = torchaudio.load(wav_path)
        waveform = waveform.unsqueeze(0)
        output_audio = {
            "waveform": waveform,
            "sample_rate": sample_rate
        }

        with open(wav_path.replace(".wav", ".txt"), "r", encoding="utf-8") as f:
            text = f.read()

        return (output_audio, text,)


NODE_CLASS_MAPPINGS = {
    "DiaTTSRun": DiaTTSRun,
    "DiaSpeakersPreview": DiaSpeakersPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DiaSpeakersPreview": "DiaTTS Speakers Preview",
    "DiaTTSRun": "DiaTTS Run",
}