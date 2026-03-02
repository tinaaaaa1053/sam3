import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 🔥 关键：无GUI模式
import matplotlib.pyplot as plt


# ==========================
# 修改这里
# ==========================
PKL_PATH = "result.pkl"
IMAGE_PATH = "result.png"
OUTPUT_PATH = "visualization_result.png"

# ==========================



def visualize_and_save(image_path, data):
    image = plt.imread(image_path)

    masks = data.get("masks", [])
    boxes = data.get("boxes", [])
    scores = data.get("scores", [])

    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    for i in range(len(masks)):
        mask = np.array(masks[i])
        plt.imshow(mask, alpha=0.4)

        if i < len(boxes):
            x, y, w, h = boxes[i]
            rect = plt.Rectangle((x, y), w, h, fill=False, linewidth=2)
            plt.gca().add_patch(rect)

        if i < len(scores):
            plt.text(x, y - 5, f"{scores[i]:.2f}", fontsize=12)

    plt.axis("off")
    plt.title("Mask Visualization")

    plt.savefig(OUTPUT_PATH, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    with open(PKL_PATH, "rb") as f:
        data = pickle.load(f)

    visualize_and_save(IMAGE_PATH, data)