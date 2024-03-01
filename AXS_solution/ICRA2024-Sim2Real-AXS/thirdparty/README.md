**Third Party Installation:**
1. activate env

```bash
conda activate your_env
```

2. install pytorch torchvision nightly (torch>=2.2.0.dev20231026)

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch-nightly -c nvidia
```

3. install thirdparty

```bash
./install.sh
```
