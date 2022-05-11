"""
    Author: Zhenyu Tang, Ziyin Zhang

    Reference: https://github.com/CMACH508/RPCL-pix2seq/blob/main/seq2png.py
"""

import cairosvg
import os
import svgwrite
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

dataset_folder = "./dataset"
save_folder = "./"

svg_folder = os.path.join(save_folder, "svg")
png_folder = os.path.join(save_folder, "png")


def get_bounds(data: np.array):
    min_x, max_x, min_y, max_y = 0, 0, 0, 0
    abs_x, abs_y = 0, 0
    for i in range(len(data)):
        x, y = float(data[i, 0]), float(data[i, 1])
        abs_x += x
        abs_y += y
        min_x = min(min_x, abs_x)
        min_y = min(min_y, abs_y)
        max_x = max(max_x, abs_x)
        max_y = max(max_y, abs_y)
    return (min_x, max_x, min_y, max_y)


def draw_strokes(data: np.array, svg_filename: str, width: int = 28, margin: float = 1.5, color: str = 'black'):
    min_x, max_x, min_y, max_y = get_bounds(data)
    if max_x - min_x > max_y - min_y:
        norm = max_x - min_x
        border_y = (norm - (max_y - min_y)) * 0.5
        border_x = 0
    else:
        norm = max_y - min_y
        border_x = (norm - (max_x - min_x)) * 0.5
        border_y = 0

    # normalize data
    norm = max(norm, 10e-6)
    scale = (width - 2 * margin) / norm
    dx = 0 - min_x + border_x
    dy = 0 - min_y + border_y

    abs_x = (0 + dx) * scale + margin
    abs_y = (0 + dy) * scale + margin

    # start converting
    dwg = svgwrite.Drawing(svg_filename, size=(width, width))
    dwg.add(dwg.rect(insert=(0, 0), size=(width, width), fill='white'))
    lift_pen = 1
    p = "M%s,%s " % (abs_x, abs_y)
    command = "m"
    for i in range(len(data)):
        if (lift_pen == 1):
            command = "m"
        elif (command != "l"):
            command = "l"
        else:
            command = ""
        x = float(data[i, 0]) * scale
        y = float(data[i, 1]) * scale
        lift_pen = data[i, 2]
        p += command + str(x) + "," + str(y) + " "
    the_color = color  # "black"
    stroke_width = 1
    dwg.add(dwg.path(p).stroke(the_color, stroke_width).fill("none"))
    dwg.save()


def svg_2_png(svg_path: str, png_path: str):
    with open(svg_path, "r") as f: svg = f.read()
    cairosvg.svg2png(bytestring=svg, write_to=png_path)
    os.remove(svg_path)


def seq_2_pic(datazip):
    zipped_files = np.load(os.path.join(dataset_folder, datazip), encoding="bytes", allow_pickle=True)
    class_svg_folder = os.path.join(svg_folder, datazip[:-4])
    class_png_folder = os.path.join(png_folder, datazip[:-4])
    for file in zipped_files:
        print(datazip, file)
        split_svg_folder = os.path.join(class_svg_folder, file)
        split_png_folder = os.path.join(class_png_folder, file)
        os.makedirs(split_svg_folder, exist_ok=True)
        os.makedirs(split_png_folder, exist_ok=True)
        for i, img_stroke in enumerate(tqdm(zipped_files[file])):
            img_svg_path = os.path.join(split_svg_folder, str(i) + ".svg")
            img_png_path = os.path.join(split_png_folder, str(i) + ".png")
            draw_strokes(img_stroke, img_svg_path)
            svg_2_png(img_svg_path, img_png_path)


if __name__ == "__main__":

    with Pool(15) as p:
        p.map(
            seq_2_pic,
            [datazip for datazip in os.listdir(dataset_folder)]
        )
