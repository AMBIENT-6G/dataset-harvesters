import re
import os

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_path = os.path.dirname(current_dir)
filename = os.path.splitext(os.path.basename(current_file_path))[0]

name = "rt-0.0-2-0.01"
filename_input = f"{current_dir}\\plots\\{name}.tex"
filename_output = f"{current_dir}\\plots\\{name}_updated.tex"

def fix_tikz_rectangles(input_file, output_file):
    pattern = re.compile(
        r'\(axis cs:([0-9\.\-]+),([0-9\.\-]+)\)\s*rectangle\s*\(axis cs:([0-9\.\-]+),([0-9\.\-]+)\)'
    )

    # pattern_zeros = re.compile(
    #     r'\\draw\[.*?\]\s*\(axis cs:([0-9\.\-]+),([0-9\.\-]+)\)\s*rectangle\s*\(axis cs:([0-9\.\-]+),(0\.0+|0)\);\n?'
    # )

    def replacer(match):
        x1, y1, x2, y2 = map(float, match.groups())
        new_y2 = y2 - y1
        return f"(axis cs:{x1},{y1}) rectangle (axis cs:{x2},{new_y2})"

    with open(input_file, "r") as f:
        content = f.read()

    content = pattern.sub(replacer, content)

    # verwijder alle matches
    # content = pattern_zeros.sub("", content)

        # 1. Verwijder rectangles met y2 = 0
    rect_pattern = re.compile(
        r'\\draw\[.*?\]\s*\(axis cs:[0-9\.\-]+,[0-9\.\-]+\)\s*rectangle\s*\(axis cs:[0-9\.\-]+,(0\.0+|0)\);\n?'
    )
    content = rect_pattern.sub("", content)

    # 2. Verwijder nodes met waarde 0
    node_pattern = re.compile(
        r'\\draw\s*\(axis cs:[0-9\.\-]+,(0\.0+|0)\)\s*node\[.*?\]\s*\{\s*0\s*\};\n?',
        re.DOTALL
    )
    content = node_pattern.sub("", content)

    with open(output_file, "w") as f:
        f.write(content)


# gebruik
fix_tikz_rectangles(filename_input, filename_output)