# app/media/video_renderer_v1.py

import os
import re
import math
import textwrap
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from moviepy.editor import (
    AudioFileClip,
    VideoClip,
    concatenate_videoclips,
)

# =========================
# CONFIG (UPGRADED)
# =========================
VIDEO_W = 1280
VIDEO_H = 720
FPS = 24

FRAME_MARGIN_X = 30
FRAME_MARGIN_TOP = 40
FRAME_MARGIN_BOTTOM = 30

WOOD_FRAME_W = 34  # 🔥 thicker frame
BOARD_RADIUS = 28

CAT_AREA_W = 260

TEXT_PADDING_X = 40
TEXT_PADDING_Y = 35

QUESTION_FONT_SIZE = 32
ANSWER_FONT_SIZE = 40
TITLE_FONT_SIZE = 34

LINE_SPACING = 26
WORD_GAP = 12

BOARD_BG = (28, 70, 45)
FRAME_COLOR = (115, 78, 42)
FRAME_EDGE = (70, 45, 20)
OUTER_BG = (218, 205, 180)

TITLE_COLOR = (0, 0, 0)
QUESTION_COLOR = (255, 235, 170)
TEXT_COLOR = (245, 245, 235)
TEXT_SHADOW = (170, 170, 160)

STICK_COLOR = (110, 80, 45)

# =========================
# HELPERS
# =========================
def pil_to_np(img):
    return np.array(img.convert("RGB"))

def load_font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except:
        return ImageFont.load_default()

def split_sentences(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())

def wrap_text(draw, text, font, max_width):
    words = text.split()
    lines = []
    current = []

    for w in words:
        test = " ".join(current + [w])
        box = draw.textbbox((0, 0), test, font=font)

        if box[2] - box[0] <= max_width:
            current.append(w)
        else:
            lines.append(current)
            current = [w]

    if current:
        lines.append(current)

    return lines

# =========================
# RENDERER
# =========================
class Renderer:
    def __init__(self):
        self.font_q = load_font(QUESTION_FONT_SIZE)
        self.font_a = load_font(ANSWER_FONT_SIZE)
        self.font_title = load_font(TITLE_FONT_SIZE)

        self.frame_x1 = FRAME_MARGIN_X
        self.frame_y1 = FRAME_MARGIN_TOP
        self.frame_x2 = VIDEO_W - FRAME_MARGIN_X
        self.frame_y2 = VIDEO_H - FRAME_MARGIN_BOTTOM

        self.board_x1 = self.frame_x1 + WOOD_FRAME_W
        self.board_y1 = self.frame_y1 + WOOD_FRAME_W
        self.board_x2 = self.frame_x2 - WOOD_FRAME_W
        self.board_y2 = self.frame_y2 - WOOD_FRAME_W

        self.cat_x = self.board_x1 + 120

        self.text_x = self.board_x1 + CAT_AREA_W + TEXT_PADDING_X
        self.text_y = self.board_y1 + TEXT_PADDING_Y

        self.text_w = self.board_x2 - self.text_x - 30

    # =========================
    # BASE
    # =========================
    def base(self):
        img = Image.new("RGB", (VIDEO_W, VIDEO_H), OUTER_BG)
        draw = ImageDraw.Draw(img)

        # frame
        draw.rounded_rectangle(
            (self.frame_x1, self.frame_y1, self.frame_x2, self.frame_y2),
            radius=BOARD_RADIUS,
            fill=FRAME_COLOR,
            outline=FRAME_EDGE,
            width=5
        )

        # board
        draw.rounded_rectangle(
            (self.board_x1, self.board_y1, self.board_x2, self.board_y2),
            radius=BOARD_RADIUS - 10,
            fill=BOARD_BG
        )

        # title inside frame
        draw.text(
            ((VIDEO_W//2)-160, self.frame_y1 + 10),
            "AI Teaching Assistant",
            font=self.font_title,
            fill=TITLE_COLOR
        )

        return img

    # =========================
    # CAT + REALISTIC STICK
    # =========================
    def draw_cat(self, draw, progress, target):

        base_x = self.cat_x
        base_y = self.board_y2 - 120

        # body
        draw.ellipse((base_x-55, base_y-85, base_x+55, base_y+70), fill=(230,180,120))

        # head
        draw.ellipse((base_x-65, base_y-150, base_x+65, base_y-40), fill=(230,180,120))

        # eyes
        draw.ellipse((base_x-30, base_y-110, base_x-10, base_y-90), fill="black")
        draw.ellipse((base_x+10, base_y-110, base_x+30, base_y-90), fill="black")

        # =========================
        # STICK (FIXED LENGTH + ROTATION)
        # =========================
        tx, ty = target

        arm_x, arm_y = base_x + 40, base_y - 40

        # angle to target
        angle = math.atan2((ty+40 - arm_y), (tx - arm_x))

        length = 140  # 🔥 fixed wooden stick length

        end_x = arm_x + length * math.cos(angle)
        end_y = arm_y + length * math.sin(angle)

        # 🔥 ensure stick is BELOW text
        end_y += 30

        draw.line([(arm_x, arm_y), (end_x, end_y)], fill=STICK_COLOR, width=6)

        return (end_x, end_y)

    # =========================
    # TEXT (CHUNK BY CHUNK)
    # =========================
    def draw_text(self, draw, question, answer, t, duration):

        # question
        q_lines = wrap_text(draw, question, self.font_q, self.text_w)

        y = self.text_y

        for line in q_lines:
            draw.text((self.text_x, y), " ".join(line), fill=QUESTION_COLOR, font=self.font_q)
            y += 42

        y += 30

        # chunk logic
        sentences = split_sentences(answer)
        chunk_index = min(len(sentences)-1, int((t/duration)*len(sentences)))

        current_chunk = sentences[chunk_index]

        lines = wrap_text(draw, current_chunk, self.font_a, self.text_w)

        words_layout = []
        wy = y

        for line in lines:
            wx = self.text_x
            for word in line:
                box = draw.textbbox((0,0), word, font=self.font_a)
                words_layout.append((word, wx, wy, box))
                wx += (box[2]-box[0]) + WORD_GAP
            wy += (box[3]-box[1]) + LINE_SPACING

        total_words = len(words_layout)
        visible = max(1, int((t/duration)*total_words))

        # draw words
        for i,(word,x,y,box) in enumerate(words_layout[:visible]):
            draw.text((x+1,y+1), word, fill=TEXT_SHADOW, font=self.font_a)
            draw.text((x,y), word, fill=TEXT_COLOR, font=self.font_a)

        # target for stick (center of current word)
        word = words_layout[min(visible-1, total_words-1)]
        cx = word[1] + (word[3][2]-word[3][0])//2
        cy = word[2] + (word[3][3]-word[3][1])//2

        return (cx, cy)

    # =========================
    # FRAME
    # =========================
    def render_frame(self, question, answer, t, duration):

        img = self.base()
        draw = ImageDraw.Draw(img)

        target = self.draw_text(draw, question, answer, t, duration)

        self.draw_cat(draw, t/duration, target)

        return img

    # =========================
    # MAIN
    # =========================
    def render(self, question, answer, output, audio=None):

        duration = 8

        if audio and os.path.exists(audio):
            a = AudioFileClip(audio)
            duration = a.duration

        def make_frame(t):
            return pil_to_np(self.render_frame(question, answer, t, duration))

        video = VideoClip(make_frame, duration=duration).set_fps(FPS)

        if audio:
            video = video.set_audio(AudioFileClip(audio))

        video.write_videofile(output, codec="libx264", audio_codec="aac")

        return output


# =========================
# ENTRY
# =========================
def render_teaching_video(question_text, answer_text, output_path, full_audio_path=None):

    r = Renderer()

    return r.render(
        question_text,
        answer_text,
        output_path,
        audio=full_audio_path
    )