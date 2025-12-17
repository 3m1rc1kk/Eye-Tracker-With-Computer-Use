# os_computer_use/utils.py
import io
from typing import Tuple
from PIL import Image, ImageDraw

def draw_big_dot(png_bytes: bytes, x: int, y: int,
                 radius: int = 12,
                 color: Tuple[int, int, int] = (255, 0, 0),
                 alpha: int = 160) -> bytes:
    """
    Ekran görüntüsünün üzerine (x,y) konumuna yarı saydam büyük bir nokta çizer.
    PNG baytlarını geri döner.
    """
    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    draw = ImageDraw.Draw(img, "RGBA")
    r = radius
    draw.ellipse((x - r, y - r, x + r, y + r),
                 fill=(color[0], color[1], color[2], alpha))
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()



def send_bbox_request(image_data, prompt):
    result = osatlas.predict(
        image=image_data,
        text_input=prompt + "\nReturn the response in the form of a bbox",
        model_id=osatlas_config["model"],
        api_name=osatlas_config["api_name"],
    )
    midpoint = extract_bbox_midpoint(result[1])
    print("BBOX: " + result[2])
    return midpoint


def extract_bbox_midpoint(bbox_response):
    match = re.search(r"<\|box_start\|>(.*?)<\|box_end\|>", bbox_response)
    inner_text = match.group(1) if match else bbox_response
    numbers = [float(num) for num in re.findall(r"\d+\.\d+|\d+", inner_text)]
    if len(numbers) == 2:
        return numbers[0], numbers[1]
    elif len(numbers) >= 4:
        return (numbers[0] + numbers[2]) // 2, (numbers[1] + numbers[3]) // 2
    else:
        return None
