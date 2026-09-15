# 构建前需准备：
#   1) Miniconda installer：文件与 Dockerfile 平级
#   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda-installer.sh
#
#   2) flash_attn 2.8.3 wheel：文件与 Dockerfile 平级
#   wget -O flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl \
#     https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
#   3) constraints.txt：文件与 Dockerfile 平级(当前代码库中已经包含)
FROM nvcr.io/nvidia/pytorch:25.06-py3

ARG CONDAENV=LimiX

ENV TZ=Asia/Shanghai

# conda 源由 conda create 的 --override-channels 指定；pip 源走下方 ENV PIP_*
COPY constraints.txt /tmp/constraints.txt

# NGC 25.03+ 用 /etc/pip/constraint.txt + PIP_CONSTRAINT 钉死基座 torch（如 2.8.0a0+nv25.6），
# 会与后续 pip install torch==2.9.1 冲突。备份后挪开，并清空 PIP_CONSTRAINT。
RUN set -eux; \
    mkdir -p /root/ngc-pip-backup; \
    if [ -n "${PIP_CONSTRAINT:-}" ] && [ -f "${PIP_CONSTRAINT}" ]; then \
      cp -a "${PIP_CONSTRAINT}" "/root/ngc-pip-backup/$(basename "${PIP_CONSTRAINT}").from-env"; \
    fi; \
    for f in /etc/pip/constraint.txt /etc/pip/constraints.txt \
             /etc/pip/constraint*.txt; do \
      [ -e "$f" ] || continue; \
      [ -f "$f" ] || continue; \
      base="$(basename "$f")"; \
      cp -a "$f" "/root/ngc-pip-backup/${base}"; \
      mv "$f" "${f}.ngc-dist"; \
    done; \
    ls -la /root/ngc-pip-backup /etc/pip 2>/dev/null || true; \
    (grep -n 'torch' /root/ngc-pip-backup/* 2>/dev/null | head -n 20) || true

ENV PIP_CONSTRAINT=

# NGC 基座 Ubuntu 源常残缺；按 VERSION_CODENAME 重写阿里云源（保留 CUDA/NVIDIA list）。
# pstree 在 psmisc 包中。失败时 dump 源 / apt-cache policy 便于 CI 排障。
# 基础镜像已经有 g++ 和 nvcc。
RUN set -eux; \
    . /etc/os-release; \
    : "${VERSION_CODENAME:?missing VERSION_CODENAME}"; \
    if [ -f /etc/apt/sources.list ]; then \
      mv /etc/apt/sources.list /etc/apt/sources.list.dist; \
    fi; \
    for f in /etc/apt/sources.list.d/ubuntu.sources \
             /etc/apt/sources.list.d/ubuntu.sources.save \
             /etc/apt/sources.list.d/debian.sources; do \
      [ -e "$f" ] && mv "$f" "$f.dist" || true; \
    done; \
    printf '%s\n' \
      "deb https://mirrors.aliyun.com/ubuntu/ ${VERSION_CODENAME} main restricted universe multiverse" \
      "deb https://mirrors.aliyun.com/ubuntu/ ${VERSION_CODENAME}-updates main restricted universe multiverse" \
      "deb https://mirrors.aliyun.com/ubuntu/ ${VERSION_CODENAME}-backports main restricted universe multiverse" \
      "deb https://mirrors.aliyun.com/ubuntu/ ${VERSION_CODENAME}-security main restricted universe multiverse" \
      > /etc/apt/sources.list; \
    apt-get update || { \
      echo '===== apt-get update failed; dumping sources ====='; \
      ls -la /etc/apt/sources.list /etc/apt/sources.list.d/ || true; \
      echo '--- /etc/apt/sources.list ---'; cat /etc/apt/sources.list || true; \
      for f in /etc/apt/sources.list.d/*; do \
        [ -f "$f" ] || continue; \
        echo "--- $f ---"; cat "$f" || true; \
      done; \
      exit 1; \
    }; \
    apt-cache policy zip | head -n 20; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        rsync locales ca-certificates \
        fonts-wqy-microhei \
        sudo tmux htop lsof zip unzip openssh-client openssh-server \
        sysstat strace procps iftop iperf3 nload tree psmisc \
        build-essential python3-dev swig tini \
      || { \
        echo '===== apt-get install failed; sample policies ====='; \
        for p in zip lsof swig psmisc htop; do echo "== $p =="; apt-cache policy "$p" || true; done; \
        exit 1; \
      }; \
    locale-gen zh_CN.UTF-8; \
    ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime; \
    echo "Asia/Shanghai" > /etc/timezone; \
    apt-get clean

ENV LANG=zh_CN.UTF-8 \
    LC_ALL=zh_CN.UTF-8 \
    LANGUAGE=zh_CN.UTF-8

# Miniconda：构建前放入上下文，构建内只 COPY + 静默安装。
COPY miniconda-installer.sh /tmp/miniconda-installer.sh
RUN set -eux; \
    bash /tmp/miniconda-installer.sh -b -p /root/miniconda3; \
    rm -f /tmp/miniconda-installer.sh

ENV PATH=/root/miniconda3/bin:$PATH

# ToS：官方可达则签署；不可达则跳过。create 顺序：清华 → 中科大 → 官方。
RUN set -eux; \
    export CONDA_REMOTE_CONNECT_TIMEOUT_SECS=15 CONDA_REMOTE_MAX_RETRIES=1 \
           CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes; \
    ( conda tos accept --override-channels \
        --channel https://repo.anaconda.com/pkgs/main \
        --channel https://repo.anaconda.com/pkgs/r ) \
    || ( echo "===== conda tos accept skipped (mirror/offline) ====="; true ); \
    ( echo "===== conda create via tuna ====="; \
      conda create -y -n "${CONDAENV}" python=3.12.7 pip \
        --override-channels \
        --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main \
        --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r ) \
    || ( echo "===== tuna failed; try ustc ====="; \
         conda env remove -y -n "${CONDAENV}" 2>/dev/null || true; \
         conda create -y -n "${CONDAENV}" python=3.12.7 pip \
           --override-channels \
           --channel https://mirrors.ustc.edu.cn/anaconda/pkgs/main \
           --channel https://mirrors.ustc.edu.cn/anaconda/pkgs/r ) \
    || ( echo "===== ustc failed; try anaconda.com ====="; \
         conda env remove -y -n "${CONDAENV}" 2>/dev/null || true; \
         conda create -y -n "${CONDAENV}" python=3.12.7 pip \
           --override-channels \
           --channel https://repo.anaconda.com/pkgs/main \
           --channel https://repo.anaconda.com/pkgs/r ); \
    conda clean --all -y

ENV PATH=/root/miniconda3/envs/${CONDAENV}/bin:$PATH \
    CONDA_DEFAULT_ENV=${CONDAENV}

# pip 源策略（本机无法访问国外源）：
#   1) cu129 的 torch/vision/audio → PIP_FIND_LINKS 阿里云 pytorch-wheels/cu129
#   2) 其余包 → PIP_INDEX_URL 阿里云 pypi
#   3) 阿里没有 → PIP_EXTRA_INDEX_URL 清华 pypi 兜底
ENV PIP_FIND_LINKS=https://mirrors.aliyun.com/pytorch-wheels/cu129/ \
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
    PIP_EXTRA_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple/ \
    PIP_TRUSTED_HOST="mirrors.aliyun.com pypi.tuna.tsinghua.edu.cn files.pythonhosted.org"

RUN pip install -c /tmp/constraints.txt setuptools setuptools-scm wheel && \
    pip install -c /tmp/constraints.txt \
        --find-links https://mirrors.aliyun.com/pytorch-wheels/cu129/ \
        torch==2.9.1+cu129 torchvision==0.24.1+cu129 torchaudio==2.9.1+cu129

# xgboost 默认会改 nccl 版本，用 xgboost-cu12 保持 nccl 12 代
RUN pip install -c /tmp/constraints.txt \
        hyperopt asposestorage catboost configspace einops gpytorch graphviz \
        huggingface_hub joblib kditransform matplotlib \
        boto3 networkx numba numpy openml pandas psutil pytest \
        pytorch_lightning pytorch_tabnet PyYAML Requests scikit-learn scipy seaborn smac \
        torchmetrics tqdm typing_extensions xgboost-cu12 coverage pytest-cov \
        python-dotenv nvtx mysql-connector-python opencv-python-headless lightgbm && \
    pip install -c /tmp/constraints.txt --force-reinstall gpustat

# 钉版本包先卸干净再装。
# 不要装 PyPI pytorch-triton（0.0.1 stub 会覆盖 triton）。
# TE/misc 可能把 numpy 拉到 2.x；numpy/pandas/datasets 必须在全部安装之后再钉死。
# triton==3.5.1 与 torch 2.9.1+cu129 配套，末尾 force-reinstall。
RUN set -eux; \
    pip uninstall -y \
        transformer_engine transformer_engine_cu12 transformer_engine_torch \
        triton pytorch-triton \
        pandas numpy \
        || true; \
    pip install -c /tmp/constraints.txt "transformer_engine==2.8.0" "transformer_engine_cu12==2.8.0"; \
    pip install -c /tmp/constraints.txt --no-build-isolation --no-binary transformer-engine-torch "transformer_engine_torch==2.8.0"; \
    pip uninstall -y pytorch-triton || true; \
    pip install -c /tmp/constraints.txt \
        --force-reinstall --no-deps \
        nvidia-nccl-cu12==2.27.5 \
        triton==3.5.1; \
    pip install -c /tmp/constraints.txt pandas==2.3.3; \
    pip install -c /tmp/constraints.txt \
        --force-reinstall --no-deps \
        numpy==1.26.4; \
    pip cache purge; \
    conda clean --all -y; \
    apt-get clean

#  pip install -c /tmp/constraints.txt datasets==4.0.0; \
# flash-attn：预编译 wheel（cu12 + torch2.9 + cxx11abiTRUE + cp312）
COPY flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl /tmp/flash_attn.whl
RUN set -eux; \
    pip install --force-reinstall --no-deps /tmp/flash_attn.whl; \
    rm -f /tmp/flash_attn.whl; \
    python -c "import flash_attn; \
assert flash_attn.__version__.startswith('2.8.3'), flash_attn.__version__; \
print('OK_FLASH_ATTN', flash_attn.__version__)"

RUN rm -rf /tmp/* /var/tmp/* /root/.cache/conda

# 钉版本冒烟（构建期验证，失败即中止）
RUN python -c "import flash_attn, triton, torch, torchvision, torchaudio, numpy, pandas; \
a = torch.__version__.split('+')[0]; \
b = torchvision.__version__.split('+')[0]; \
c = torchaudio.__version__.split('+')[0]; \
assert (a, b, c) == ('2.9.1', '0.24.1', '2.9.1'), (a, b, c); \
assert '+cu129' in torch.__version__, torch.__version__; \
assert '+cu129' in torchvision.__version__, torchvision.__version__; \
assert '+cu129' in torchaudio.__version__, torchaudio.__version__; \
assert numpy.__version__ == '1.26.4', numpy.__version__; \
assert pandas.__version__ == '2.3.3', pandas.__version__; \
assert triton.__version__ == '3.5.1', getattr(triton, '__version__', triton); \
assert flash_attn.__version__.startswith('2.8.3'), flash_attn.__version__; \
print('OK_PINS', torch.__version__, torchvision.__version__, torchaudio.__version__, numpy.__version__, pandas.__version__, 'triton', triton.__version__, 'flash_attn', flash_attn.__version__)"



# 交互进容器读 ~/.bashrc；login 壳读 /etc/profile。配置只写一份，两处 source。
RUN cat > /etc/container-env.sh <<EOF
# sourced by /etc/profile and ~/.bashrc
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ${CONDAENV}
export PATH=/root/miniconda3/envs/${CONDAENV}/bin:\$PATH
EOF

RUN set -eux; \
    touch /root/.bashrc; \
    grep -qF '/etc/container-env.sh' /root/.bashrc \
      || echo '[ -f /etc/container-env.sh ] && . /etc/container-env.sh' >> /root/.bashrc; \
    grep -qF '/etc/container-env.sh' /etc/profile \
      || echo '[ -f /etc/container-env.sh ] && . /etc/container-env.sh' >> /etc/profile

CMD ["/bin/bash", "-l"]
