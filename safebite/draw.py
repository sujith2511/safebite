"""Draw allergen highlights on the packet photo: red = contains, yellow = may contain."""
from PIL import Image, ImageDraw, ImageFont

RED, YELLOW = (220, 38, 38), (234, 179, 8)


def _font(size: int):
    for path in ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def annotate(img: Image.Image, marks: list[dict], block_box: list[float] | None = None) -> Image.Image:
    """marks: [{"box": [x1,y1,x2,y2], "color": "red"|"yellow", "label": "whey -> milk"}]"""
    out = img.convert("RGB").copy()
    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    width = max(3, out.width // 400)
    font = _font(max(12, out.width // 95))
    if block_box:
        d.rectangle(block_box, outline=(59, 130, 246, 200), width=max(2, width // 2))
    for m in marks:
        if not m.get("box"):
            continue
        rgb = RED if m["color"] == "red" else YELLOW
        d.rectangle(m["box"], fill=rgb + (60,), outline=rgb + (255,), width=width)
        if m.get("label"):
            x, y = m["box"][0], max(0, m["box"][1] - font.size - 6)
            tw = d.textlength(m["label"], font=font)
            d.rectangle([x, y, x + tw + 8, y + font.size + 6], fill=rgb + (235,))
            d.text((x + 4, y + 2), m["label"], fill=(255, 255, 255, 255) if m["color"] == "red"
                   else (0, 0, 0, 255), font=font)
    return Image.alpha_composite(out.convert("RGBA"), overlay).convert("RGB")
