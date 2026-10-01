import os
import math
import re
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

BOARD_MARGIN_X = 55
BOARD_MARGIN_TOP = 45
BOARD_MARGIN_BOTTOM = 45

TEXT_PADDING_X = 40
TEXT_PADDING_Y = 35
LINE_SPACING = 18
WORD_GAP = 12

DEFAULT_FONT_SIZE = 34
TITLE_FONT_SIZE = 40
QUESTION_FONT_SIZE = 27

BOARD_RADIUS = 24
BOARD_BORDER = 14

BG_COLOR = (205, 190, 155)
BOARD_COLOR = (25, 70, 42)
BOARD_BORDER_COLOR = (120, 85, 45)

TEXT_COLOR = (243, 243, 235)
TITLE_COLOR = (250, 250, 240)
QUESTION_COLOR = (255, 240, 180)
SUBTLE_TEXT_COLOR = (225, 232, 220)
HIGHLIGHT_COLOR = (255, 224, 92)

STICK_COLOR = (238, 228, 210)
HAND_COLOR = (232, 196, 160)
HAND_OUTLINE = (120, 80, 50)
POINTER_TIP_COLOR = (255, 245, 210)

QUESTION_WRAP = 52
ANSWER_WRAP_PIXELS_MARGIN = 2 * TEXT_PADDING_X

# Hand base position (right side)
HAND_BASE_X = VIDEO_W - 145
HAND_BASE_Y = VIDEO_H - 205


# =========================================================
# HELPERS
# =========================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text.strip())
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
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ])

    for fp in candidates:
        try:
            return ImageFont.truetype(fp, size=size)
        except Exception:
            continue

    return ImageFont.load_default()


def pil_to_np(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))


def rounded_rect(draw: ImageDraw.ImageDraw, xy, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


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
        "such", "very", "more", "most", "across", "without", "where"
    }

    tokens = re.findall(r"\b[A-Za-z][A-Za-z0-9\-]{2,}\b", sentence)
    freq = {}
    for t in tokens:
        k = t.lower()
        if k not in stop_words:
            freq[k] = freq.get(k, 0) + 1

    ranked = sorted(freq.items(), key=lambda x: (-x[1], -len(x[0]), x[0]))
    return [w for w, _ in ranked[:max_keywords]]


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def ease_in_out(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 3 * t * t - 2 * t * t * t


def wrap_words_to_lines(
    words: List[str],
    font,
    max_width: int,
    draw: ImageDraw.ImageDraw
) -> List[List[str]]:
    lines = []
    current = []

    for word in words:
        candidate = current + [word]
        candidate_text = " ".join(candidate)
        bbox = draw.textbbox((0, 0), candidate_text, font=font)
        width = bbox[2] - bbox[0]

        if width <= max_width or not current:
            current.append(word)
        else:
            lines.append(current)
            current = [word]

    if current:
        lines.append(current)

    return lines


def get_word_layout(
    text: str,
    draw: ImageDraw.ImageDraw,
    font,
    max_width: int,
    start_x: int,
    start_y: int,
    line_spacing: int
):
    raw_words = text.split()
    lines = wrap_words_to_lines(raw_words, font, max_width, draw)

    layout = []
    y = start_y
    line_index = 0

    for line_words in lines:
        x = start_x
        line_bbox = draw.textbbox((0, 0), "Ag", font=font)
        line_h = line_bbox[3] - line_bbox[1]

        for word in line_words:
            bbox = draw.textbbox((0, 0), word, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]

            layout.append({
                "word": word,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "line": line_index,
                "center": (x + w // 2, y + h // 2),
            })
            x += w + WORD_GAP

        y += line_h + line_spacing
        line_index += 1

    total_height = y - start_y
    return layout, total_height


# =========================================================
# RENDERER
# =========================================================

class ChalkTeachingRenderer:
    def __init__(
        self,
        width: int = VIDEO_W,
        height: int = VIDEO_H,
        fps: int = FPS,
        font_path: Optional[str] = None,
        title_font_path: Optional[str] = None,
    ):
        self.width = width
        self.height = height
        self.fps = fps

        self.font = load_font(font_path, DEFAULT_FONT_SIZE)
        self.title_font = load_font(title_font_path or font_path, TITLE_FONT_SIZE)
        self.question_font = load_font(font_path, QUESTION_FONT_SIZE)
        self.small_font = load_font(font_path, 26)

        self.board_x1 = BOARD_MARGIN_X
        self.board_y1 = BOARD_MARGIN_TOP
        self.board_x2 = self.width - BOARD_MARGIN_X
        self.board_y2 = self.height - BOARD_MARGIN_BOTTOM

        self.text_area_x1 = self.board_x1 + TEXT_PADDING_X
        self.text_area_y1 = self.board_y1 + TEXT_PADDING_Y + 22
        self.text_area_x2 = self.board_x2 - TEXT_PADDING_X - 60
        self.text_area_y2 = self.board_y2 - TEXT_PADDING_Y

        self.text_area_w = self.text_area_x2 - self.text_area_x1
        self.text_area_h = self.text_area_y2 - self.text_area_y1

    # -----------------------------------------------------
    # BASE
    # -----------------------------------------------------
    def _base_canvas(self, title: str = "AI TEACHING ASSISTANT") -> Image.Image:
        img = Image.new("RGB", (self.width, self.height), BG_COLOR)
        draw = ImageDraw.Draw(img)

        rounded_rect(
            draw,
            (self.board_x1, self.board_y1, self.board_x2, self.board_y2),
            radius=BOARD_RADIUS,
            fill=BOARD_BORDER_COLOR,
            outline=(70, 45, 20),
            width=3
        )

        inner_pad = BOARD_BORDER
        rounded_rect(
            draw,
            (
                self.board_x1 + inner_pad,
                self.board_y1 + inner_pad,
                self.board_x2 - inner_pad,
                self.board_y2 - inner_pad,
            ),
            radius=BOARD_RADIUS - 6,
            fill=BOARD_COLOR,
            outline=(235, 225, 215),
            width=2
        )

        if title:
            bbox = draw.textbbox((0, 0), title, font=self.title_font)
            tw = bbox[2] - bbox[0]
            tx = max(20, (self.width - tw) // 2)
            ty = 8
            draw.text((tx, ty), title, font=self.title_font, fill=TITLE_COLOR)

        return img

    # -----------------------------------------------------
    # HAND + STICK
    # -----------------------------------------------------
    def _draw_hand_with_stick(
        self,
        draw: ImageDraw.ImageDraw,
        hand_center: Tuple[float, float],
        target_point: Tuple[float, float]
    ) -> None:
        hx, hy = hand_center
        tx, ty = target_point

        # stick line
        draw.line([(hx, hy), (tx, ty)], fill=STICK_COLOR, width=6)

        # pointer tip
        r = 5
        draw.ellipse((tx - r, ty - r, tx + r, ty + r), fill=POINTER_TIP_COLOR, outline=HIGHLIGHT_COLOR)

        # simple hand shape
        palm_w = 34
        palm_h = 28
        px1 = hx - palm_w // 2
        py1 = hy - palm_h // 2
        px2 = hx + palm_w // 2
        py2 = hy + palm_h // 2

        rounded_rect(
            draw,
            (px1, py1, px2, py2),
            radius=10,
            fill=HAND_COLOR,
            outline=HAND_OUTLINE,
            width=2
        )

        # fingers
        finger_w = 6
        finger_h = 16
        start_x = px1 + 5
        for i in range(4):
            fx1 = start_x + i * 7
            fy1 = py1 - 8
            fx2 = fx1 + finger_w
            fy2 = fy1 + finger_h
            rounded_rect(
                draw,
                (fx1, fy1, fx2, fy2),
                radius=3,
                fill=HAND_COLOR,
                outline=HAND_OUTLINE,
                width=1
            )

    def _compute_hand_position(
        self,
        active_target: Tuple[int, int],
        progress: float
    ) -> Tuple[float, float]:
        # hand starts from fixed base and moves smoothly toward target
        eased = ease_in_out(progress)
        hx = lerp(HAND_BASE_X, active_target[0] + 110, eased)
        hy = lerp(HAND_BASE_Y, active_target[1] - 65, eased)
        return hx, hy

    # -----------------------------------------------------
    # TEXT DRAWING
    # -----------------------------------------------------
    def _draw_scene_words(
        self,
        base_img: Image.Image,
        question_text: str,
        answer_text: str,
        progress: float,
        highlight_words: Optional[List[str]] = None,
    ) -> Image.Image:
        img = base_img.copy()
        draw = ImageDraw.Draw(img)

        # question
        wrapped_question = textwrap.fill(f"Question: {question_text}", width=QUESTION_WRAP)
        draw.multiline_text(
            (self.text_area_x1, self.text_area_y1 - 10),
            wrapped_question,
            font=self.question_font,
            fill=QUESTION_COLOR,
            spacing=7
        )

        # measure question height
        q_bbox = draw.multiline_textbbox(
            (self.text_area_x1, self.text_area_y1 - 10),
            wrapped_question,
            font=self.question_font,
            spacing=7
        )
        q_height = q_bbox[3] - q_bbox[1]

        draw.text(
            (self.text_area_x1, self.text_area_y1 + q_height + 15),
            "Explanation",
            font=self.small_font,
            fill=SUBTLE_TEXT_COLOR
        )

        answer_start_y = self.text_area_y1 + q_height + 55

        words_layout, _ = get_word_layout(
            text=answer_text,
            draw=draw,
            font=self.font,
            max_width=self.text_area_w,
            start_x=self.text_area_x1,
            start_y=answer_start_y,
            line_spacing=LINE_SPACING
        )

        total_words = len(words_layout)
        if total_words == 0:
            return img

        exact_index = progress * total_words
        visible_words = min(total_words, max(1, math.ceil(exact_index)))
        highlight_set = normalize_keywords(highlight_words)

        active_center = words_layout[visible_words - 1]["center"]

        # draw visible words
        for idx, item in enumerate(words_layout[:visible_words]):
            word = item["word"]
            x = item["x"]
            y = item["y"]
            w = item["w"]
            h = item["h"]

            is_highlight = clean_word(word) in highlight_set
            if is_highlight:
                rounded_rect(
                    draw,
                    (x - 6, y - 4, x + w + 6, y + h + 4),
                    radius=8,
                    fill=(110, 95, 20),
                    outline=HIGHLIGHT_COLOR,
                    width=2
                )

            draw.text((x + 1, y + 1), word, font=self.font, fill=(200, 200, 190))
            draw.text((x, y), word, font=self.font, fill=TEXT_COLOR)

        # smooth hand movement between words
        hand_x, hand_y = self._compute_hand_position(active_center, progress)
        self._draw_hand_with_stick(draw, (hand_x, hand_y), active_center)

        return img

    # -----------------------------------------------------
    # CLIP BUILD
    # -----------------------------------------------------
    def build_sentence_clip(
        self,
        question_text: str,
        sentence: str,
        duration: float,
        highlight_words: Optional[List[str]] = None,
        title: str = "AI TEACHING ASSISTANT"
    ) -> VideoClip:
        duration = max(0.8, float(duration))
        base = self._base_canvas(title=title)

        def make_frame(t: float):
            progress = min(1.0, max(0.0, t / duration))
            frame_img = self._draw_scene_words(
                base_img=base,
                question_text=question_text,
                answer_text=sentence,
                progress=progress,
                highlight_words=highlight_words
            )
            return pil_to_np(frame_img)

        return VideoClip(make_frame, duration=duration).set_fps(self.fps)

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
        teacher_title: str = "AI TEACHING ASSISTANT",
        codec: str = "libx264",
        audio_codec: str = "aac"
    ) -> str:
        ensure_dir(os.path.dirname(output_path) or ".")

        sentences = split_sentences(answer_text)
        if not sentences:
            raise ValueError("answer_text is empty after sentence split.")

        clips = []

        # Best sync mode: one audio per sentence
        if sentence_audio_paths:
            if len(sentence_audio_paths) != len(sentences):
                raise ValueError(
                    f"sentence_audio_paths count ({len(sentence_audio_paths)}) does not match sentence count ({len(sentences)})."
                )

            for i, sentence in enumerate(sentences):
                audio_path = sentence_audio_paths[i]
                if not os.path.exists(audio_path):
                    raise FileNotFoundError(f"Sentence audio not found: {audio_path}")

                audio_clip = AudioFileClip(audio_path)
                duration = max(0.8, float(audio_clip.duration))

                keywords = None
                if highlight_map and i in highlight_map:
                    keywords = highlight_map[i]
                elif auto_highlight:
                    keywords = extract_keywords(sentence)

                scene_clip = self.build_sentence_clip(
                    question_text=question_text,
                    sentence=sentence,
                    duration=duration,
                    highlight_words=keywords,
                    title=teacher_title
                ).set_audio(audio_clip)

                clips.append(scene_clip)

            final_video = concatenate_videoclips(clips, method="compose")

        else:
            full_audio_clip = None
            total_audio_duration = None

            if full_audio_path:
                if not os.path.exists(full_audio_path):
                    raise FileNotFoundError(f"Full audio file not found: {full_audio_path}")
                full_audio_clip = AudioFileClip(full_audio_path)
                total_audio_duration = float(full_audio_clip.duration)

            word_counts = [max(1, len(s.split())) for s in sentences]
            total_words = sum(word_counts)

            for i, sentence in enumerate(sentences):
                if total_audio_duration is not None:
                    duration = total_audio_duration * (word_counts[i] / total_words)
                else:
                    duration = max(2.0, word_counts[i] * 0.38)

                keywords = None
                if highlight_map and i in highlight_map:
                    keywords = highlight_map[i]
                elif auto_highlight:
                    keywords = extract_keywords(sentence)

                scene_clip = self.build_sentence_clip(
                    question_text=question_text,
                    sentence=sentence,
                    duration=duration,
                    highlight_words=keywords,
                    title=teacher_title
                )
                clips.append(scene_clip)

            video_without_audio = concatenate_videoclips(clips, method="compose")

            if full_audio_clip is not None:
                final_video = video_without_audio.set_audio(full_audio_clip)
                final_video = final_video.set_duration(full_audio_clip.duration)
            else:
                final_video = video_without_audio

        final_video.write_videofile(
            output_path,
            fps=self.fps,
            codec=codec,
            audio_codec=audio_codec,
            threads=4,
            preset="medium"
        )

        try:
            final_video.close()
        except Exception:
            pass

        for c in clips:
            try:
                c.close()
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
    renderer = ChalkTeachingRenderer(
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
    sample_question = "What is packet switching?"
    sample_answer = (
        "Packet switching is a method where data is split into small packets that travel independently across the network. "
        "Each packet is routed by switches to its destination without requiring a continuous connection. "
        "This approach allows multiple communications to share network resources efficiently by reusing links dynamically. "
        "It avoids the inefficiency of circuit switching, which needs dedicated paths for the entire session."
    )

    render_teaching_video(
        question_text=sample_question,
        answer_text=sample_answer,
        output_path="demo_teaching_video.mp4",
        full_audio_path=None,
    )