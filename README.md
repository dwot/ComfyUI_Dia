[中文](README-CN.md)|[English](README.md)

# Dia's ComfyUI Node

![](https://github.com/billwuhao/ComfyUI_Dia/blob/main/images/2025-04-24_08-56-13.png)

Text-to-speech, voice cloning, generating highly realistic dialogue in one go. When cloning a voice, the `clone-prompt-text` needs to be placed before `gen-text`; only the audio of `gen-text` will be generated.

Supported vocal tags include `(laughs), (clears throat), (sighs), (gasps), (coughs), (singing), (sings), (mumbles), (beep), (groans), (sniffs), (claps), (screams), (inhales), (exhales), (applause), (burps), (humming), (sneezes), (chuckle), (whistles)`.

## 📣 Updates

[2025-04-24]⚒️: Released v1.0.0.

## Installation

```
cd ComfyUI/custom_nodes
git clone https://github.com/billwuhao/ComfyUI_Dia.git
cd ComfyUI_Dia
pip install -r requirements.txt

# python_embeded
./python_embeded/python.exe -m pip install -r requirements.txt
```

## Model Download

- [Dia-1.6B](https://huggingface.co/nari-labs/Dia-1.6B/tree/main): Download the entire directory and place it in the `ComfyUI/models/TTS` directory.
- [weights.pth](https://github.com/descriptinc/descript-audio-codec/releases/download/0.0.1/weights.pth): Download and rename it to `weights_44khz_8kbps_0.0.1.pth`. Place it in the `ComfyUI/models/TTS/DAC.speech.v1.0` directory.

## Acknowledgements

[dia](https://github.com/nari-labs/dia)
