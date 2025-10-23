import re
import urllib.parse

# 输入输出文件名
input_file = "Update_Version_3.md"
output_file = "Update_Version_3_github.md"

# 读取原始 Markdown 文件
with open(input_file, "r", encoding="utf-8") as f:
    content = f.read()

# 正则表达式匹配 $$...$$ 块公式
pattern = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)

def convert_latex_to_github(match):
    """把 LaTeX 块公式转换为 GitHub 可显示的 math image"""
    latex_code = match.group(1).strip()
    encoded = urllib.parse.quote(latex_code)
    github_math = (
        f'<p align="center">\n'
        f'  <img src="https://render.githubusercontent.com/render/math?math={encoded}" />\n'
        f'</p>\n'
    )
    return github_math

# 执行替换
converted_content = re.sub(pattern, convert_latex_to_github, content)

# 保存新文件
with open(output_file, "w", encoding="utf-8") as f:
    f.write(converted_content)

print(f"✅ Conversion complete! Saved to: {output_file}")
