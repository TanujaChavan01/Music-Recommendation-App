from PIL import Image, ImageDraw, ImageFont

width, height = 800, 150
header = Image.new("RGB", (width, height), color=(255, 255, 255))

for x in range(width):
    for y in range(height):
        gradient_color = (x // 4, y // 6, 255 - x // 4)
        header.putpixel((x, y), gradient_color)

draw = ImageDraw.Draw(header)

title_font = ImageFont.truetype("arial.ttf", 36)
subtitle_font = ImageFont.truetype("arial.ttf", 20)

title_text = "Emotion-Based Music Recommendation"
title_bbox = title_font.getbbox(title_text)
title_width = title_bbox[2] - title_bbox[0]
title_height = title_bbox[3] - title_bbox[1]
draw.text(((width - title_width) / 2, 30), title_text, fill="white", font=title_font)

header.save("header.jpg")
print("Header image saved as header.jpg")

