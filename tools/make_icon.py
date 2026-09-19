"""Draw the application icon (magnifier over a pixel grid) -> assets/fiasuite.ico"""
from pathlib import Path

from PIL import Image, ImageDraw

S = 256
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
d.rounded_rectangle((8, 8, S - 8, S - 8), 48, fill=(24, 25, 27, 255))
colors = [(62, 99, 221), (229, 72, 77), (245, 217, 10), (70, 167, 88)]
for i in range(4):
    for j in range(4):
        c = colors[(i * 3 + j * 5) % 4] if (i, j) in {(1, 1), (1, 2), (2, 1)} else (46, 49, 53)
        d.rounded_rectangle((40 + j * 36, 40 + i * 36, 70 + j * 36, 70 + i * 36), 6, fill=c + (255,))
d.ellipse((96, 96, 196, 196), outline=(237, 238, 240, 255), width=14)
d.line((182, 182, 226, 226), fill=(237, 238, 240, 255), width=20)
out = Path(__file__).resolve().parent.parent / "assets" / "fiasuite.ico"
img.save(out, sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])
print(out)
