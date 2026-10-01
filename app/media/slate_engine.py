from PIL import Image, ImageDraw, ImageFont
import textwrap

W, H = 1280, 720

def create_frame(text: str):
    img = Image.new("RGB", (W, H), (10, 10, 18))  # dark clean bg
    draw = ImageDraw.Draw(img)

    # default font (no extra install). If you want custom font later, we can add.
    font = ImageFont.load_default()

    # wrap text
    lines = textwrap.wrap(text, width=60)
    lines = lines[:10]  # avoid overflow

    y = 140
    for line in lines:
        draw.text((100, y), line, fill=(245, 245, 245), font=font)
        y += 35

    # footer style
    draw.text((100, 650), "Teaching Assistant • Auto video", fill=(170, 170, 170), font=font)
    return img