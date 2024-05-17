# 基本イメージ
FROM mcr.microsoft.com/vscode/devcontainers/python:3.11

# Poetryのインストール
RUN curl -sSL https://install.python-poetry.org | python3 -

# PATH設定
ENV PATH="/home/vscode/.local/bin:$PATH"

# 作業ディレクトリ
WORKDIR /workspace
