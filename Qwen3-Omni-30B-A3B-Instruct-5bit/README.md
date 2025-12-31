---
license: other
license_name: apache-2.0
language:
- en
tags:
- multimodal
- mlx
library_name: transformers
pipeline_tag: any-to-any
---

# mlx-community/Qwen3-Omni-30B-A3B-Instruct-5bit
This model was converted to MLX format from [`Qwen/Qwen3-Omni-30B-A3B-Instruct`]() using mlx-vlm version **0.3.10**.
Refer to the [original model card](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) for more details on the model.
## Use with mlx

```bash
pip install -U mlx-vlm
```

```bash
python -m mlx_vlm.generate --model mlx-community/Qwen3-Omni-30B-A3B-Instruct-5bit --max-tokens 100 --temperature 0.0 --prompt "Describe this image." --image <path_to_image>
```
