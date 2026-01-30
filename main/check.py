import numpy as np
import os

file_path = '/home/xyp/sx/SUTrack/data/absent.txt'


def try_read():
    print(f"--- 正在分析文件: {file_path} ---")




    # 3. 尝试作为二进制 uint8 读取 (1字节整数，0/1 常用)
    try:
        data = np.fromfile(file_path, dtype=np.uint8)
        print("✅ 识别成功：该文件是二进制 uint8 格式")
        print("数据预览:", data[:1000])
        return
    except:
        pass

    print("❌ 无法识别格式，请确认文件是否损坏或提供乱码截图。")


if __name__ == "__main__":
    if os.path.exists(file_path):
        try_read()
    else:
        print("找不到 absent.txt 文件，请检查路径。")