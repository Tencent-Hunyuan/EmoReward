
# Download the pretrained checkpoints:

The base-model layout follows the upstream [SAGE-GRPO checkpoint guide](https://github.com/Tencent-Hunyuan/SAGE-GRPO/blob/master/checkpoints-download.md). Run the commands below from the EmoReward repository root. Emotional-reward provider weights are documented separately in [REWARD_SUBPACKAGES.md](./REWARD_SUBPACKAGES.md).

First, make sure you have installed the huggingface CLI and modelscope CLI.

```bash
pip install -U "huggingface_hub[cli]"
pip install modelscope
```


### Download the pretrained DiT and VAE checkpoints:
```bash
hf download tencent/HunyuanVideo-1.5 --local-dir ./ckpts
```

### Downloading TextEncoders and Vision Encoders

HunyuanVideo uses an MLLM and a byT5 as text encoders.

* **MLLM**

    HunyuanVideo can be integrated with different MLLMs (including HunyuanMLLM and other open-source MLLM models). 

    At this stage, we have not yet released the latest HunyuanMLLM. We recommend the users in community to use an open-source alternative, such as Qwen2.5-VL-7B-Instruct provided by Qwen Team, which can be downloaded by the following command:
    ```bash
    hf download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ./ckpts/text_encoder/llm
    ```

* **ByT5 encoder**

    We use [Glyph-SDXL-v2](https://modelscope.cn/models/AI-ModelScope/Glyph-SDXL-v2) as our [byT5](https://github.com/google-research/byt5) encoder, which can be downloaded by the following command:

    ```bash
    hf download google/byt5-small --local-dir ./ckpts/text_encoder/byt5-small
    modelscope download --model AI-ModelScope/Glyph-SDXL-v2 --local_dir ./ckpts/text_encoder/Glyph-SDXL-v2
    ```
    You can also manually download the checkpoints from [this link](https://modelscope.cn/models/AI-ModelScope/Glyph-SDXL-v2/files) and place them in the text_encoder folder like:
    ```
    ckpts
    ├── text_encoder
    │   ├── Glyph-SDXL-v2
    │   │   ├── assets
    │   │   │   ├── color_idx.json
    │   │   │   ├── multilingual_10-lang_idx.json
    │   │   │   └── ...
    │   │   └── checkpoints
    │   │       ├── byt5_model.pt
    │   │       └── ...
    │   └─  ...
    └─  ...
    ```

* **Vision Encoder**
    We use Siglip as the vision encoder. To download this model, you need to request access to the [FLUX.1-Redux-dev model](https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev) on the Hugging Face website. After your request is approved, use your personal Hugging Face access token to download the model as follows (please replace `<your_hf_token>` with your actual Hugging Face access token):
    ```bash
    hf download black-forest-labs/FLUX.1-Redux-dev --local-dir ./ckpts/vision_encoder/siglip --token <your_hf_token>
    ```


<details>

<summary>💡Tips for using hf/huggingface-cli (network problem)</summary>

##### 1. Using HF-Mirror

If you encounter slow download speeds in China, you can try a mirror to speed up the download process:

```shell
HF_ENDPOINT=https://hf-mirror.com hf download tencent/HunyuanVideo-1.5 --local-dir ./ckpts
```

##### 2. Resume Download

`huggingface-cli` supports resuming downloads. If the download is interrupted, you can just rerun the download 
command to resume the download process.

Note: If an `No such file or directory: 'ckpts/.huggingface/.gitignore.lock'` like error occurs during the download 
process, you can ignore the error and rerun the download command.

</details>


## VideoReward / VideoAlign

Download the general video-reward checkpoint into `./ckpts/VideoReward`:

```bash
bash download_weights.sh
```

The VideoAlign scorer also accepts `VIDEOALIGN_MODEL_PATH` to reuse an existing checkpoint.

## Optional FLUX.1-schnell for HPSv3/COHP

The bundled HPSv3/COHP image-generation utility has an optional FLUX.1-schnell pipeline.
It is separate from the HunyuanVideo training checkpoints described in the
[SAGE-GRPO guide](https://github.com/Tencent-Hunyuan/SAGE-GRPO/blob/master/checkpoints-download.md).
Its standard local path is `./ckpts/FLUX.1-schnell`; set `FLUX_SCHNELL_MODEL_PATH`
to reuse another local directory or a Hugging Face model ID.

Review the access conditions on the [official model page](https://huggingface.co/black-forest-labs/FLUX.1-schnell)
and sign in with `hf auth login` if required, then download from the repository root:

```bash
hf download black-forest-labs/FLUX.1-schnell --local-dir ./ckpts/FLUX.1-schnell
```

Alternatively, download VideoReward and this optional checkpoint together:

```bash
DOWNLOAD_FLUX_SCHNELL=1 bash download_weights.sh
```

To use an existing local copy, set the override before launching the HPSv3/COHP utility:

```bash
export FLUX_SCHNELL_MODEL_PATH=/path/to/FLUX.1-schnell
```

Model weights remain excluded from Git.
