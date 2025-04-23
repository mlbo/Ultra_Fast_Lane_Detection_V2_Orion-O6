#!/bin/bash

# 检查是否已安装Git和SSH工具
if ! command -v git &> /dev/null; then
    echo "Git未安装，请先安装Git。"
    exit 1
fi

if ! command -v ssh &> /dev/null; then
    echo "SSH工具未安装，请先安装OpenSSH客户端。"
    exit 1
fi

# 设置Git用户名和邮箱
echo "正在配置Git用户名和邮箱..."
read -p "请输入您的Git用户名（例如：Your Name）: " GIT_USER_NAME
read -p "请输入您的Git邮箱（例如：your_email@example.com）: " GIT_USER_EMAIL

# 全局设置Git用户名和邮箱 [[2]]
git config --global user.name "$GIT_USER_NAME"
git config --global user.email "$GIT_USER_EMAIL"

# 验证Git配置是否成功
CONFIGURED_USER_NAME=$(git config --global user.name)
CONFIGURED_USER_EMAIL=$(git config --global user.email)

if [[ "$CONFIGURED_USER_NAME" == "$GIT_USER_NAME" && "$CONFIGURED_USER_EMAIL" == "$GIT_USER_EMAIL" ]]; then
    echo "Git用户名和邮箱配置成功！"
    echo "用户名: $CONFIGURED_USER_NAME"
    echo "邮箱: $CONFIGURED_USER_EMAIL"
else
    echo "Git配置失败，请检查错误信息。"
    exit 1
fi

# 设置SSH密钥文件路径
SSH_KEY_PATH="$HOME/.ssh/id_rsa_github"

# 检查是否已存在SSH密钥
if [[ -f "$SSH_KEY_PATH" ]]; then
    echo "SSH密钥已存在: $SSH_KEY_PATH"
else
    # 生成新的SSH密钥
    echo "正在生成新的SSH密钥..."
    ssh-keygen -t rsa -b 4096 -C "$GIT_USER_EMAIL" -f "$SSH_KEY_PATH" -N ""
    if [[ $? -ne 0 ]]; then
        echo "生成SSH密钥失败，请检查错误信息。"
        exit 1
    fi
    echo "SSH密钥生成成功: $SSH_KEY_PATH"
fi

# 将SSH密钥添加到ssh-agent
echo "启动ssh-agent并添加密钥..."
eval "$(ssh-agent -s)"
ssh-add "$SSH_KEY_PATH"

# 显示公钥内容
SSH_PUBLIC_KEY=$(cat "${SSH_KEY_PATH}.pub")
echo "您的SSH公钥为:"
echo "$SSH_PUBLIC_KEY"

# 提示用户将公钥添加到GitHub
echo "请将上述公钥复制，并按照以下步骤添加到GitHub账户中:"
echo "1. 登录GitHub网站。"
echo "2. 点击右上角的头像，选择'Settings'。"
echo "3. 在左侧菜单中点击'SSH and GPG keys'。"
echo "4. 点击'New SSH key'按钮，在标题中填写描述，例如'My Laptop'。"
echo "5. 将公钥粘贴到密钥框中，并点击'Add SSH key'。" [[7]]

# 测试SSH连接
echo "正在测试与GitHub的SSH连接..."
ssh -T git@github.com
if [[ $? -eq 0 ]]; then
    echo "SSH连接测试成功！您可以正常使用SSH与GitHub交互。" [[2]]
else
    echo "SSH连接测试失败，请检查配置步骤。" [[4]]
fi