import os


def rename_files():
    # 获取当前目录下的所有文件
    files = os.listdir('.')

    # 初始化文件名计数器
    count = 1

    # 遍历所有文件
    for file in files:
        # 检查是否是文件而不是目录
        if os.path.isfile(file):
            # 获取文件名和扩展名
            filename, ext = os.path.splitext(file)
            if ext == '.jpg':
            # 构建新的文件名，使用字符串格式化来保持数字长度一致
                new_filename = f"picture_{count:02d}{ext}"

                # 重命名文件
                os.rename(file, new_filename)
                print(f"Renamed {file} to {new_filename}")

                # 增加计数器
                count += 1


# 调用函数
rename_files()
