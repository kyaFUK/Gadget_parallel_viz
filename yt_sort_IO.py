import os

# ファイルサイズを人間に読みやすい形式に変換する関数
def format_size(size):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"  # サイズが非常に大きい場合

# サイズ順でファイルを取得する関数
def get_file_sizes_sorted(src):
    try:
        files_with_size = []
        
        # ディレクトリ内のすべてのファイルを再帰的に取得
        for root, _, files in os.walk(src):
            for file in files:
                file_path = os.path.join(root, file)
                size = os.path.getsize(file_path)  # バイト単位のファイルサイズを取得
                human_readable_size = format_size(size)  # 人間に読みやすい形式に変換
                files_with_size.append((file_path, size, human_readable_size))
        
        # サイズで降順にソート
        files_with_size.sort(key=lambda x: x[1], reverse=True)
        
        # ファイルパスとフォーマット済みのサイズをリストとして返す
        return [(file_path, human_readable_size) for file_path, _, human_readable_size in files_with_size]
    except Exception as e:
        print(f"Error: {e}")
        return []
    
def size_in_bytes(size_str):
    units = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}
    
    # 数字部分と単位部分を分割
    size, unit = size_str.split()
    unit = unit.upper()  # 単位を大文字にする

    # 単位をバイト数に変換
    return float(size) * units[unit]

src = "/data/hp120286/u10158/MWGenIC9216_InitTurb/results/snapshots00001"
# 使用例
# Get file sizes and sort them by size using 'ls -lhS'
files_with_size_human = get_file_sizes_sorted(src)
print(files_with_size_human[:3], "\n\n")

# Convert sizes to bytes for easier calculations and sorting
files_with_size = [(file, size_in_bytes(size)) for file, size in files_with_size_human]
print(files_with_size[:3], "\n\n")

# Sort files by size (largest first)
sorted_files_with_size = sorted(files_with_size, key=lambda x: x[1], reverse=True)
print(sorted_files_with_size[:3], "\n\n")


# Extract only filenames (sorted by size)
sorted_file_list = [file for file, size in sorted_files_with_size]
print(sorted_file_list[:3], "\n\n")