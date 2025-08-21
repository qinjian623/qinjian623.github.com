#!/usr/bin/env bash
# replace_html_blocks.sh
# 将 Org 文件中 #+BEGIN_HTML/# +END_HTML 替换为 #+BEGIN_EXPORT html/# +END_EXPORT

SRC_DIR="${1:-./}"  # 默认源目录

find "$SRC_DIR" -type f -name "*.org" | while read -r file; do
    echo "Processing $file"
     # 使用 sed 替换 BEGIN/END 标签，允许前导空格
    sed -i \
        -e 's/BEGIN_HTML/BEGIN_EXPORT\ html/' \
        -e 's/END_HTML/END_EXPORT/' \
        "$file"
    head -n 6 $file
done

echo "Replacement done."
