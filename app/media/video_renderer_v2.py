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

# =========================================================
# CONFIG
# =========================================================

VIDEO_W = 1280
VIDEO_H = 720
FPS = 24

FRAME_MARGIN_X = 42
FRAME_MARGIN_TOP = 52
FRAME_MARGIN_BOTTOM = 42

WOOD_FRAME_W = 26
BOARD_RADIUS = 28

CAT_AREA_W = 265

TEXT_PADDING_X = 34
TEXT_PADDING_Y = 28
QUESTION_LINE_SPACING = 9
ANSWER_LINE_SPACING = 18
WORD_GAP = 11

QUESTION_FONT_SIZE = 28
ANSWER_FONT_SIZE = 34
TITLE_FONT_SIZE = 28
SECTION_FONT_SIZE = 22

BOARD_BG = (28, 66, 44)
BOARD_SHADOW = (18, 48, 32)
FRAME_COLOR = (121, 84, 48)
FRAME_EDGE = (83, 55, 28)
OUTER_BG = (218, 205, 180)

TITLE_COLOR = (15, 15, 15)
QUESTION_COLOR = (255, 235, 170)
TEXT_COLOR = (244, 244, 236)
TEXT_SHADOW = (190, 195, 188)
SECTION_COLOR = (225, 232, 220)

CAT_FUR = (228, 182, 124)
CAT_FUR_DARK = (196, 146, 92)
CAT_INNER_EAR = (240, 170, 170)
CAT_EYE = (38, 38, 38)
CAT_NOSE = (160, 100, 105)
CAT_MOUTH = (90, 60, 60)
CAT_OUTLINE = (100, 72, 40)
CAT_BELLY = (242, 220, 192)
CAT_CHEEK = (238, 178, 164)
STICK_COLOR = (114, 82, 49)
PAW_PAD = (204, 146, 146)

CHALK_COLORS = [
    (244, 244, 236),
    (255, 223, 120),
    (181, 226, 255),
    (196, 255, 196),
    (255, 186, 214),
]

QUESTION_WRAP_WIDTH = 48
TITLE_TEXT = "AI Teaching Assistant"

# =========================================================
# HELPERS
# =========================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def pil_to_np(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def load_font(font_path: Optional[str], size: int):
    candidates = []
    if font_path and os.path.exists(font_path):
        candidates.append(font_path)

    candidates.extend([
        "arial.ttf",
        "Arial.ttf",
        "DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ])

    for fp in candidates:
        try:
            return ImageFont.truetype(fp, size=size)
        except Exception:
            continue

    return ImageFont.load_default()


def rounded_rect(draw: ImageDraw.ImageDraw, xy, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def ease_in_out(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 3 * t * t - 2 * t * t * t


def clean_word(word: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "", word.lower()).strip()


def normalize_keywords(words: Optional[List[str]]) -> set:
    if not words:
        return set()
    return {clean_word(w) for w in words if clean_word(w)}


def extract_keywords(sentence: str, max_keywords: int = 4) -> List[str]:
    stop_words = {
        "the", "is", "a", "an", "and", "or", "of", "to", "in", "for", "with",
        "that", "this", "it", "on", "as", "are", "be", "by", "from", "at",
        "can", "has", "have", "will", "was", "were", "their", "its", "into",
        "than", "then", "which", "used", "use", "using", "each", "also",
        "such", "very", "more", "most", "across", "without", "where", "what",
        "how", "does", "actually", "work", "simple", "way", "understand"
    }
    tokens = re.findall(r"\b[A-Za-z][A-Za-z0-9\-]{2,}\b", sentence)
    freq = {}
    for tok in tokens:
        key = tok.lower()
        if key not in stop_words:
            freq[key] = freq.get(key, 0) + 1
    ranked = sorted(freq.items(), key=lambda x: (-x[1], -len(x[0]), x[0]))
    return [w for w, _ in ranked[:max_keywords]]


def split_text_pages(
    text: str,
    draw: ImageDraw.ImageDraw,
    font,
    max_width: int,
    max_height: int,
    line_spacing: int,
) -> List[List[List[str]]]:
    words = text.split()
    pages: List[List[List[str]]] = []
    current_page: List[List[str]] = []
    current_line: List[str] = []

    line_bbox = draw.textbbox((0, 0), "Ag", font=font)
    line_h = line_bbox[3] - line_bbox[1]
    max_lines = max(1, max_height // (line_h + line_spacing))

    def line_width(line_words: List[str]) -> int:
        if not line_words:
            return 0
        txt = " ".join(line_words)
        bbox = draw.textbbox((0, 0), txt, font=font)
        return bbox[2] - bbox[0]

    for word in words:
        candidate = current_line + [word]
        if not current_line or line_width(candidate) <= max_width:
            current_line.append(word)
        else:
            current_page.append(current_line)
            current_line = [word]

            if len(current_page) >= max_lines:
                pages.append(current_page)
                current_page = []

    if current_line:
        current_page.append(current_line)

    if current_page:
        pages.append(current_page)

    return pages


def count_words_in_page(page: List[List[str]]) -> int:
    return sum(len(line) for line in page)


def build_color_map(keywords: List[str]) -> dict:
    color_map = {}
    for i, kw in enumerate(keywords):
        color_map[kw.lower()] = CHALK_COLORS[(i + 1) % len(CHALK_COLORS)]
    return color_map


# =========================================================
# CAT TEACHER RENDERER
# =========================================================

class CatTeacherRenderer:
    def __init__(
        self,
        width: int = VIDEO_W,
        height: int = VIDEO_H,
        fps: int = FPS,
        font_path: Optional[str] = None,
    ):
        self.width = width
        self.height = height
        self.fps = fps

        self.question_font = load_font(font_path, QUESTION_FONT_SIZE)
        self.answer_font = load_font(font_path, ANSWER_FONT_SIZE)
        self.title_font = load_font(font_path, TITLE_FONT_SIZE)
        self.section_font = load_font(font_path, SECTION_FONT_SIZE)

        self.frame_x1 = FRAME_MARGIN_X
        self.frame_y1 = FRAME_MARGIN_TOP
        self.frame_x2 = self.width - FRAME_MARGIN_X
        self.frame_y2 = self.height - FRAME_MARGIN_BOTTOM

        self.board_x1 = self.frame_x1 + WOOD_FRAME_W
        self.board_y1 = self.frame_y1 + WOOD_FRAME_W
        self.board_x2 = self.frame_x2 - WOOD_FRAME_W
        self.board_y2 = self.frame_y2 - WOOD_FRAME_W

        self.cat_area_x1 = self.board_x1 + 18
        self.cat_area_x2 = self.cat_area_x1 + CAT_AREA_W

        self.text_area_x1 = self.cat_area_x2 + TEXT_PADDING_X
        self.text_area_y1 = self.board_y1 + TEXT_PADDING_Y + 6
        self.text_area_x2 = self.board_x2 - TEXT_PADDING_X
        self.text_area_y2 = self.board_y2 - TEXT_PADDING_Y

        self.text_area_w = self.text_area_x2 - self.text_area_x1
        self.text_area_h = self.text_area_y2 - self.text_area_y1

    # -----------------------------------------------------
    # BASE SCENE
    # -----------------------------------------------------
    def _base_canvas(self) -> Image.Image:
        img = Image.new("RGB", (self.width, self.height), OUTER_BG)
        draw = ImageDraw.Draw(img)

        rounded_rect(
            draw,
            (self.frame_x1, self.frame_y1, self.frame_x2, self.frame_y2),
            radius=BOARD_RADIUS,
            fill=FRAME_COLOR,
            outline=FRAME_EDGE,
            width=4,
        )

        rounded_rect(
            draw,
            (self.board_x1, self.board_y1, self.board_x2, self.board_y2),
            radius=BOARD_RADIUS - 8,
            fill=BOARD_BG,
            outline=(225, 215, 200),
            width=2,
        )

        # subtle board depth
        draw.rounded_rectangle(
            (self.board_x1 + 8, self.board_y1 + 8, self.board_x2 - 8, self.board_y2 - 8),
            radius=BOARD_RADIUS - 12,
            outline=BOARD_SHADOW,
            width=2,
        )

        # title on top wood frame
        title_bbox = draw.textbbox((0, 0), TITLE_TEXT, font=self.title_font)
        title_w = title_bbox[2] - title_bbox[0]
        title_x = (self.width - title_w) // 2
        title_y = self.frame_y1 + 9
        draw.text((title_x, title_y), TITLE_TEXT, font=self.title_font, fill=TITLE_COLOR)

        return img

    # -----------------------------------------------------
    # CAT
    # -----------------------------------------------------
    def _draw_cat_teacher(
        self,
        draw: ImageDraw.ImageDraw,
        progress: float,
        target_point: Tuple[int, int],
    ) -> Tuple[float, float]:
        t = max(0.0, min(1.0, progress))

        base_x = self.cat_area_x1 + 118
        bounce = math.sin(progress * math.pi * 2.0) * 3.5
        body_y = self.board_y2 - 120 + bounce
        head_y = body_y - 172

        # idle motion
        tail_wave = math.sin(progress * math.pi * 2.4) * 10
        blink = abs(math.sin(progress * math.pi * 2.2))
        blink_amt = 1.0 if blink > 0.96 else 0.0

        # body
        draw.ellipse(
            (base_x - 62, body_y - 92, base_x + 58, body_y + 70),
            fill=CAT_FUR,
            outline=CAT_OUTLINE,
            width=3,
        )
        draw.ellipse(
            (base_x - 35, body_y - 42, base_x + 32, body_y + 52),
            fill=CAT_BELLY,
            outline=CAT_OUTLINE,
            width=2,
        )

        # head
        draw.ellipse(
            (base_x - 72, head_y - 62, base_x + 72, head_y + 68),
            fill=CAT_FUR,
            outline=CAT_OUTLINE,
            width=3,
        )

        # ears
        draw.polygon(
            [(base_x - 48, head_y - 18), (base_x - 78, head_y - 82), (base_x - 16, head_y - 54)],
            fill=CAT_FUR,
            outline=CAT_OUTLINE,
        )
        draw.polygon(
            [(base_x + 45, head_y - 18), (base_x + 75, head_y - 82), (base_x + 15, head_y - 54)],
            fill=CAT_FUR,
            outline=CAT_OUTLINE,
        )
        draw.polygon(
            [(base_x - 47, head_y - 29), (base_x - 66, head_y - 69), (base_x - 26, head_y - 49)],
            fill=CAT_INNER_EAR,
        )
        draw.polygon(
            [(base_x + 44, head_y - 29), (base_x + 64, head_y - 69), (base_x + 24, head_y - 49)],
            fill=CAT_INNER_EAR,
        )

        # cheeks
        draw.ellipse((base_x - 56, head_y + 5, base_x - 20, head_y + 36), fill=CAT_CHEEK)
        draw.ellipse((base_x + 20, head_y + 5, base_x + 56, head_y + 36), fill=CAT_CHEEK)

        # eyes
        eye_y = head_y - 4
        if blink_amt:
            draw.line((base_x - 32, eye_y, base_x - 12, eye_y), fill=CAT_EYE, width=3)
            draw.line((base_x + 12, eye_y, base_x + 32, eye_y), fill=CAT_EYE, width=3)
        else:
            draw.ellipse((base_x - 36, eye_y - 10, base_x - 12, eye_y + 13), fill=(255, 255, 255), outline=CAT_EYE)
            draw.ellipse((base_x + 12, eye_y - 10, base_x + 36, eye_y + 13), fill=(255, 255, 255), outline=CAT_EYE)
            draw.ellipse((base_x - 28, eye_y - 2, base_x - 18, eye_y + 10), fill=CAT_EYE)
            draw.ellipse((base_x + 20, eye_y - 2, base_x + 30, eye_y + 10), fill=CAT_EYE)

        # nose and mouth
        draw.polygon(
            [(base_x, head_y + 18), (base_x - 8, head_y + 28), (base_x + 8, head_y + 28)],
            fill=CAT_NOSE,
            outline=CAT_MOUTH,
        )
        draw.line((base_x, head_y + 28, base_x, head_y + 37), fill=CAT_MOUTH, width=2)
        draw.arc((base_x - 15, head_y + 32, base_x, head_y + 49), 10, 170, fill=CAT_MOUTH, width=2)
        draw.arc((base_x, head_y + 32, base_x + 15, head_y + 49), 10, 170, fill=CAT_MOUTH, width=2)

        # whiskers
        for offset in (-1, 1):
            side = -1 if offset == -1 else 1
            sx = base_x + side * 18
            draw.line((sx, head_y + 26, sx + side * 34, head_y + 18), fill=CAT_MOUTH, width=2)
            draw.line((sx, head_y + 30, sx + side * 36, head_y + 30), fill=CAT_MOUTH, width=2)
            draw.line((sx, head_y + 34, sx + side * 34, head_y + 42), fill=CAT_MOUTH, width=2)

        # legs
        draw.rounded_rectangle((base_x - 42, body_y + 38, base_x - 16, body_y + 104), radius=10, fill=CAT_FUR_DARK, outline=CAT_OUTLINE, width=2)
        draw.rounded_rectangle((base_x + 12, body_y + 38, base_x + 38, body_y + 104), radius=10, fill=CAT_FUR_DARK, outline=CAT_OUTLINE, width=2)
        draw.ellipse((base_x - 48, body_y + 92, base_x - 8, body_y + 116), fill=CAT_BELLY, outline=CAT_OUTLINE, width=2)
        draw.ellipse((base_x + 6, body_y + 92, base_x + 46, body_y + 116), fill=CAT_BELLY, outline=CAT_OUTLINE, width=2)

        # tail
        tail_points = [
            (base_x + 48, body_y + 20),
            (base_x + 100 + tail_wave * 0.45, body_y - 5),
            (base_x + 108 + tail_wave, body_y - 48),
            (base_x + 78 + tail_wave * 0.55, body_y - 78),
        ]
        draw.line(tail_points, fill=CAT_FUR_DARK, width=16)
        draw.line(tail_points, fill=CAT_OUTLINE, width=3)

        # left relaxed paw
        draw.rounded_rectangle((base_x - 78, body_y - 16, base_x - 48, body_y + 45), radius=11, fill=CAT_FUR, outline=CAT_OUTLINE, width=2)
        draw.ellipse((base_x - 82, body_y + 34, base_x - 44, body_y + 56), fill=PAW_PAD, outline=CAT_OUTLINE, width=2)

        # right teaching paw / stick
        tx, ty = target_point
        arm_start = (base_x + 48, body_y - 28)
        arm_progress = ease_in_out(t)
        paw_x = lerp(arm_start[0] + 20, tx - 72, arm_progress)
        paw_y = lerp(arm_start[1] + 40, ty + 48, arm_progress)

        draw.line([arm_start, (paw_x, paw_y)], fill=CAT_FUR_DARK, width=18)
        draw.line([arm_start, (paw_x, paw_y)], fill=CAT_OUTLINE, width=3)

        paw_box = (paw_x - 16, paw_y - 14, paw_x + 16, paw_y + 16)
        draw.ellipse(paw_box, fill=CAT_FUR, outline=CAT_OUTLINE, width=2)

        stick_tip_x = lerp(paw_x + 18, tx, 0.93)
        stick_tip_y = lerp(paw_y - 2, ty, 0.93)
        draw.line([(paw_x + 8, paw_y - 1), (stick_tip_x, stick_tip_y)], fill=STICK_COLOR, width=6)
        draw.ellipse((stick_tip_x - 4, stick_tip_y - 4, stick_tip_x + 4, stick_tip_y + 4), fill=(247, 238, 210), outline=STICK_COLOR)

        return paw_x, paw_y

    # -----------------------------------------------------
    # TEXT / PAGE
    # -----------------------------------------------------
    def _draw_question_block(self, draw: ImageDraw.ImageDraw, question_text: str) -> Tuple[int, int]:
        wrapped_question = textwrap.fill(question_text, width=QUESTION_WRAP_WIDTH)
        draw.multiline_text(
            (self.text_area_x1, self.text_area_y1),
            wrapped_question,
            font=self.question_font,
            fill=QUESTION_COLOR,
            spacing=QUESTION_LINE_SPACING,
        )
        bbox = draw.multiline_textbbox(
            (self.text_area_x1, self.text_area_y1),
            wrapped_question,
            font=self.question_font,
            spacing=QUESTION_LINE_SPACING,
        )
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    def _draw_page_words(
        self,
        draw: ImageDraw.ImageDraw,
        page_lines: List[List[str]],
        visible_words: int,
        start_y: int,
        color_map: dict,
    ) -> Tuple[Tuple[int, int], int]:
        active_center = (self.text_area_x1 + 20, start_y + 20)
        count = 0
        line_bbox = draw.textbbox((0, 0), "Ag", font=self.answer_font)
        line_h = line_bbox[3] - line_bbox[1]
        y = start_y

        for line in page_lines:
            x = self.text_area_x1
            for word in line:
                bbox = draw.textbbox((0, 0), word, font=self.answer_font)
                word_w = bbox[2] - bbox[0]
                word_h = bbox[3] - bbox[1]

                if count < visible_words:
                    color = color_map.get(clean_word(word), TEXT_COLOR)
                    draw.text((x + 1, y + 1), word, font=self.answer_font, fill=TEXT_SHADOW)
                    draw.text((x, y), word, font=self.answer_font, fill=color)
                    active_center = (x + word_w // 2, y + word_h // 2)
                count += 1
                x += word_w + WORD_GAP
            y += line_h + ANSWER_LINE_SPACING

        return active_center, count

    def _build_pages_and_timing(
        self,
        question_text: str,
        answer_text: str,
        total_duration: Optional[float],
    ):
        temp = self._base_canvas()
        draw = ImageDraw.Draw(temp)

        _, q_h = self._draw_question_block(draw, question_text)
        answer_start_y = self.text_area_y1 + q_h + 34
        available_h = self.text_area_y2 - answer_start_y

        pages = split_text_pages(
            answer_text,
            draw,
            self.answer_font,
            self.text_area_w,
            available_h,
            ANSWER_LINE_SPACING,
        )

        if not pages:
            raise ValueError("No answer pages could be created.")

        page_word_counts = [count_words_in_page(p) for p in pages]
        total_words = max(1, sum(page_word_counts))

        if total_duration is not None:
            durations = [max(1.2, total_duration * (w / total_words)) for w in page_word_counts]
            total = sum(durations)
            if abs(total - total_duration) > 0.01:
                scale = total_duration / total
                durations = [d * scale for d in durations]
        else:
            durations = [max(2.0, w * 0.35) for w in page_word_counts]

        return pages, durations, answer_start_y

    # -----------------------------------------------------
    # SINGLE FRAME
    # -----------------------------------------------------
    def _render_page_frame(
        self,
        question_text: str,
        page_lines: List[List[str]],
        local_t: float,
        page_duration: float,
        answer_start_y: int,
        color_map: dict,
        page_index: int,
        total_pages: int,
    ) -> Image.Image:
        img = self._base_canvas()
        draw = ImageDraw.Draw(img)

        _, q_h = self._draw_question_block(draw, question_text)

        # subtle page indicator if multiple pages
        if total_pages > 1:
            label = f"part {page_index + 1}/{total_pages}"
            lb = draw.textbbox((0, 0), label, font=self.section_font)
            draw.text(
                (self.text_area_x2 - (lb[2] - lb[0]), self.text_area_y1 + q_h + 4),
                label,
                font=self.section_font,
                fill=SECTION_COLOR,
            )

        progress = max(0.0, min(1.0, local_t / max(0.001, page_duration)))
        total_page_words = count_words_in_page(page_lines)
        visible_words = min(total_page_words, max(1, math.ceil(progress * total_page_words)))

        active_center, _ = self._draw_page_words(
            draw=draw,
            page_lines=page_lines,
            visible_words=visible_words,
            start_y=answer_start_y,
            color_map=color_map,
        )

        self._draw_cat_teacher(draw, progress=progress, target_point=active_center)
        return img

    # -----------------------------------------------------
    # MAIN RENDER
    # -----------------------------------------------------
    def render(
        self,
        question_text: str,
        answer_text: str,
        output_path: str,
        full_audio_path: Optional[str] = None,
        sentence_audio_paths: Optional[List[str]] = None,
        auto_highlight: bool = True,
        highlight_map: Optional[dict[int, List[str]]] = None,
        codec: str = "libx264",
        audio_codec: str = "aac",
    ) -> str:
        ensure_dir(os.path.dirname(output_path) or ".")

        if sentence_audio_paths:
            raise NotImplementedError("This renderer is optimized for full_audio_path mode.")

        full_audio_clip = None
        total_audio_duration = None

        if full_audio_path:
            if not os.path.exists(full_audio_path):
                raise FileNotFoundError(f"Full audio file not found: {full_audio_path}")
            full_audio_clip = AudioFileClip(full_audio_path)
            total_audio_duration = float(full_audio_clip.duration)

        keywords = extract_keywords(answer_text) if auto_highlight else []
        color_map = build_color_map(keywords)

        pages, durations, answer_start_y = self._build_pages_and_timing(
            question_text=question_text,
            answer_text=answer_text,
            total_duration=total_audio_duration,
        )

        clips = []

        for idx, page_lines in enumerate(pages):
            page_duration = durations[idx]

            def make_frame_factory(page_lines=page_lines, idx=idx, page_duration=page_duration):
                def make_frame(t: float):
                    frame_img = self._render_page_frame(
                        question_text=question_text,
                        page_lines=page_lines,
                        local_t=t,
                        page_duration=page_duration,
                        answer_start_y=answer_start_y,
                        color_map=color_map,
                        page_index=idx,
                        total_pages=len(pages),
                    )
                    return pil_to_np(frame_img)
                return make_frame

            clip = VideoClip(make_frame_factory(), duration=page_duration).set_fps(self.fps)
            clips.append(clip)

        final_video = concatenate_videoclips(clips, method="compose")

        if full_audio_clip is not None:
            final_video = final_video.set_audio(full_audio_clip)
            final_video = final_video.set_duration(full_audio_clip.duration)

        final_video.write_videofile(
            output_path,
            fps=self.fps,
            codec=codec,
            audio_codec=audio_codec,
            threads=4,
            preset="medium",
        )

        try:
            final_video.close()
        except Exception:
            pass

        if full_audio_clip is not None:
            try:
                full_audio_clip.close()
            except Exception:
                pass

        for clip in clips:
            try:
                clip.close()
            except Exception:
                pass

        return output_path


# =========================================================
# PUBLIC FUNCTION
# =========================================================

def render_teaching_video(
    question_text: str,
    answer_text: str,
    output_path: str,
    full_audio_path: Optional[str] = None,
    sentence_audio_paths: Optional[List[str]] = None,
    avatar_path: Optional[str] = None,
    font_path: Optional[str] = None,
) -> str:
    renderer = CatTeacherRenderer(
        width=VIDEO_W,
        height=VIDEO_H,
        fps=FPS,
        font_path=font_path,
    )

    return renderer.render(
        question_text=question_text,
        answer_text=answer_text,
        output_path=output_path,
        full_audio_path=full_audio_path,
        sentence_audio_paths=sentence_audio_paths,
    )


# =========================================================
# OPTIONAL TEST
# =========================================================

if __name__ == "__main__":
    sample_question = "How does packet switching actually work in a network?"
    sample_answer = (
        "Packet switching splits data into small packets that travel independently across the network. "
        "Each packet is forwarded by switches toward the destination without needing a dedicated connection. "
        "This allows multiple users to share network resources efficiently. "
        "At the destination, the packets are reassembled into the original message."
    )

    render_teaching_video(
        question_text=sample_question,
        answer_text=sample_answer,
        output_path="demo_cat_teacher_video.mp4",
        full_audio_path=None,
    )