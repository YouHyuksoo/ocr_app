import os
import itertools
import zipfile
import random
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter
import numpy as np
import yaml


def generate_combinations(digits, length):
    return ["".join(p) for p in itertools.product(digits, repeat=length)]


def apply_noise(image, amount=25):
    np_img = np.array(image).astype(np.int16)
    noise = np.random.randint(-amount, amount + 1, np_img.shape, dtype=np.int16)
    noisy_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)


def make_image(text, config, font_path="arial.ttf", distorted=False):
    img_w, img_h = config["image_size"]
    box_w, box_h = config["digit_box_size"]
    spacing = config["digit_spacing"]
    bg = config["background"]
    fg = config["text_color"]
    random_pos = config.get("random_position", False)
    use_random_bg = config.get("use_random_bg", False)
    apply_blur = config.get("apply_blur", False)
    apply_noise_texture = config.get("apply_noise", False)

    if use_random_bg:
        bg_color = tuple(random.randint(100, 255) for _ in range(3))
        img = Image.new("RGB", (img_w, img_h), color=bg_color)
    else:
        img = Image.new("RGB", (img_w, img_h), color=bg)

    draw = ImageDraw.Draw(img)

    font_size = box_h
    font = ImageFont.truetype(font_path, font_size)
    while True:
        size = font.getbbox("0")
        if size[3] - size[1] <= box_h:
            break
        font_size -= 1
        font = ImageFont.truetype(font_path, font_size)

    # 실제 텍스트 크기를 측정해 전체 문자열의 실제 폭 계산
    text_width = 0
    text_bboxes = []
    for char in text:
        bbox = font.getbbox(char)
        w = bbox[2] - bbox[0]
        text_width += w
        text_bboxes.append((char, bbox, w))
    total_spacing = spacing * (len(text) - 1)
    total_width = text_width + total_spacing

    start_x = (
        random.randint(0, img_w - total_width)
        if random_pos
        else (img_w - total_width) // 2
    )
    y = random.randint(0, img_h - box_h) if random_pos else (img_h - box_h) // 2

    label_data = []
    x_cursor = start_x

    for char, bbox, w in text_bboxes:
        h = bbox[3] - bbox[1]
        text_x = x_cursor
        text_y = y + (box_h - h) // 2 - bbox[1]  # adjust for ascent
        draw.text((text_x, text_y), char, font=font, fill=fg)

        x1 = text_x
        y1 = text_y + bbox[1]  # real top
        x2 = text_x + w
        y2 = text_y + h + bbox[1]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1 - 20), str(char), fill="red")

        x_center = (x1 + x2) / 2 / img_w
        y_center = (y1 + y2) / 2 / img_h
        width = (x2 - x1) / img_w
        height = (y2 - y1) / img_h
        class_id = int(char)
        label_data.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )

        x_cursor += w + spacing

    if distorted:
        angle = random.randint(-15, 15)
        img = img.rotate(
            angle, expand=True, fillcolor=bg if not use_random_bg else bg_color
        )
        img = ImageOps.fit(img, (img_w, img_h))

    if apply_blur:
        img = img.filter(ImageFilter.GaussianBlur(radius=1.2))

    if apply_noise_texture:
        img = apply_noise(img)

    return img, label_data


def generate_dataset(config, out_dir, val_split=0.1):  # val_split을 0.1(10%)로 설정
    """데이터셋을 생성하고 train/val로 분할합니다.

    Args:
        config (dict): 데이터셋 생성 설정
        out_dir (str): 출력 디렉토리
        val_split (float): 검증 데이터셋 비율 (기본값: 0.1 = 10%)
    """
    # 기본 디렉토리 구조 생성
    for split in ["train", "val"]:
        for subdir in ["images", "labels"]:
            os.makedirs(os.path.join(out_dir, subdir, split), exist_ok=True)

    # data.yaml 파일 생성
    yaml_content = {
        "train": "./images/train",
        "val": "./images/val",
        "nc": len(config["digits"]),
        "names": config["digits"],
    }

    with open(os.path.join(out_dir, "data.yaml"), "w") as f:
        yaml.dump(yaml_content, f)

    total_count = 0
    temp_images = []  # 생성된 이미지 임시 저장

    # 이미지 생성
    for _ in range(100):  # 임의의 수량, 필요에 따라 조정
        text = "".join(random.choices(config["digits"], k=config["length"]))
        img, label_info = make_image(text, config)

        temp_images.append((img, label_info, text))
        total_count += 1

    # train/val 분할
    random.shuffle(temp_images)
    val_size = int(len(temp_images) * val_split)
    train_images = temp_images[val_size:]
    val_images = temp_images[:val_size]

    # train 데이터 저장
    for idx, (img, label_info, text) in enumerate(train_images):
        img_path = os.path.join(out_dir, "images", "train", f"img_{idx:04d}.jpg")
        label_path = os.path.join(out_dir, "labels", "train", f"img_{idx:04d}.txt")

        img.save(img_path)
        with open(label_path, "w") as f:
            for label in label_info:
                f.write(f"{label}\n")

    # val 데이터 저장
    for idx, (img, label_info, text) in enumerate(val_images):
        img_path = os.path.join(out_dir, "images", "val", f"img_{idx:04d}.jpg")
        label_path = os.path.join(out_dir, "labels", "val", f"img_{idx:04d}.txt")

        img.save(img_path)
        with open(label_path, "w") as f:
            for label in label_info:
                f.write(f"{label}\n")

    return out_dir, total_count


def zip_dataset_folder(output_dir):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, output_dir)
                zipf.write(full_path, arcname=rel_path)
    zip_buffer.seek(0)
    return zip_buffer
