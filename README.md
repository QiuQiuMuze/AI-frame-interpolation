# AI-frame-interpolation

使用深度学习模型（例如 RIFE / FILM / DAIN 等）来对两张图之间进行补帧操作。

## 方案 B：基于 rife-ncnn-vulkan 的高清补帧流程

本仓库提供了一个简单的 Python 命令行包装器，方便使用 `rife-ncnn-vulkan` 可执行文件对两张静态图像之间进行 AI 补帧。

### 环境准备

1. 从 [nihui/rife-ncnn-vulkan](https://github.com/nihui/rife-ncnn-vulkan/releases) 下载与你平台相符的可执行文件，并解压到某个目录中。
2. 确保系统中安装了 Python 3.9+。脚本只依赖标准库，因此无需额外的第三方包。

### 使用方法

假设已经有 `frame0.png` 和 `frame1.png` 两张图像，以及解压后的二进制 `./rife-ncnn-vulkan`，可以使用以下命令生成中间帧：

```bash
python scripts/interpolate_pair.py \
    frame0.png frame1.png output.png \
    --binary /path/to/rife-ncnn-vulkan \
    --model rife-v4.6 \
    --time 0.5
```

主要参数说明：

- `--time`：插值时间点，范围 `[0, 1]`，默认 `0.5` 表示生成两帧的正中间帧。
- `--model`：选择 `rife-ncnn-vulkan` 附带的模型（可选）。如果省略则使用可执行文件默认模型。
- `--scale`：设置内部缩放倍率，默认为 `1.0`。
- `--tta`：启用测试时增强（Test-Time Augmentation），在追求更高画质时可开启。
- `--uhd`：针对超高清分辨率（4K/8K）场景启用 UHD 模式。
- `--gpu-id`：指定使用的 GPU 编号。
- `--threads`：限制 CPU 线程数量。
- `--extra-args`：向底层可执行文件透传未被覆盖的其他参数。

脚本内部会调用外部的 `rife-ncnn-vulkan` 工具完成 AI 补帧，因此可充分利用其高清模型与 Vulkan 加速能力。你也可以扩展脚本，将其融入到自己的工作流或图形界面中。

### 其他思路

- 如果你希望在纯 Python 环境下运行 RIFE 原始 PyTorch 模型，可以参考 [hzwer/Practical-RIFE](https://github.com/hzwer/Practical-RIFE) 获取源码与权重。
- 谷歌的 [FILM](https://github.com/google-research/frame-interpolation) 以及 DAIN 等模型也能以类似方式集成，只需在脚本中替换成对应的推理入口即可。
