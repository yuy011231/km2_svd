# Python 3.11 イメージを使用
FROM mcr.microsoft.com/vscode/devcontainers/python:3.11

# Poetryのインストール
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/opt/poetry python3 - && \
    ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry

# Poetryのパスを環境変数に追加
ENV PATH="/opt/poetry/bin:$PATH"

# 作業ディレクトリ
WORKDIR /workspace

# コンテナの作成時にPoetryのインストールを確認
RUN poetry --version