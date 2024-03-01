#!/bin/bash

# 指定包含子包的目录路径
target_directory="."

# 获取目录下所有子目录（子包）
subpackages=$(find "$target_directory" -maxdepth 3 -type d)

# 遍历每个子包的目录
for package in $subpackages; do
    # 检查子包目录中是否存在 setup.py 文件
    setup_path="$package/setup.py"
    if [ -f "$setup_path" ]; then
        # 使用 pip 执行安装
        pip install -e "$package"
    else
        echo "Warning: No setup.py found in $package. Skipping ..."
    fi
done

