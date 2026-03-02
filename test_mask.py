import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2

# ==========================
# 修改这里
# ==========================
PKL_PATH = "result.pkl"
IMAGE_PATH = "result.png"
# ==========================


def load_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data


def visualize_results(image_path, data, score_threshold=0.0):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found. Check IMAGE_PATH.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    masks = data.get("masks", [])
    boxes = data.get("boxes", [])
    scores = data.get("scores", [])

    print("========== 检测信息 ==========")
    print("检测到目标数量:", len(masks))
    print("Scores:", scores)
    print("==============================")

    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    for i in range(len(masks)):
        score = scores[i] if i < len(scores) else None

        if score is not None and score < score_threshold:
            continue

        mask = np.array(masks[i])

        # 显示 mask（半透明）
        plt.imshow(mask, alpha=0.4)

        # 画 box
        if i < len(boxes):
            x, y, w, h = boxes[i]
            rect = plt.Rectangle(
                (x, y),
                w,
                h,
                fill=False,
                linewidth=2
            )
            plt.gca().add_patch(rect)

        # 显示 score
        if score is not None:
            plt.text(
                x,
                y - 5,
                f"{score:.2f}",
                fontsize=12
            )

    plt.title("Mask Visualization")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    data = load_pkl(PKL_PATH)
    print("PKL Keys:", data.keys())

    visualize_results(
        IMAGE_PATH,
        data,
        score_threshold=0.3  # 可调节阈值
    )
    