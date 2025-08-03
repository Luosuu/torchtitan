FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel

# Base packages
RUN apt-get update && apt-get install -y \
    curl git zsh unzip build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Install oh-my-zsh (non-interactive)
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

# Set up nvm environment
ENV NVM_DIR=/root/.nvm
RUN mkdir -p "$NVM_DIR" \
    && curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash \
    && . "$NVM_DIR/nvm.sh" \
    && nvm install node \
    && nvm use node

# Add nvm to zsh and bash startup files
RUN echo 'export NVM_DIR="$HOME/.nvm"' >> ~/.zshrc \
    && echo '[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"' >> ~/.zshrc \
    && echo 'export NVM_DIR="$HOME/.nvm"' >> ~/.bashrc \
    && echo '[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"' >> ~/.bashrc

# Default to zsh
SHELL ["/usr/bin/zsh", "-c"]
