# 这是一个示例 Python 脚本。
import os

import cv2

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。

from algorithms import get_algorithm


def resize2(img, dest_size):
    h, w = img.shape[:2]
    scale = dest_size / max(h,w)
    w = int(w * scale + 0.5)
    h = int(h * scale + 0.5)
    img = cv2.resize(img, (w,h))
    return scale, img

# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    algorithm = get_algorithm("lama")()
    algorithm.prepare()

    DIR = r"E:\virtualmachine\shared\images\tattoo\tatoo_test"
    DEST = r"E:\virtualmachine\shared\images\tattoo\results"
    os.makedirs(DEST, exist_ok=True)
    for f in os.listdir(DIR):
        if "_mask" in f:
            continue
        fname, suffix = os.path.splitext(f)
        mask_f = f"{fname}_mask{suffix}"
        mask_f = os.path.join(DIR, mask_f)
        if not os.path.isfile(mask_f):
            continue
        img = cv2.imread(f"{DIR}/{f}")
        mask = cv2.imread(mask_f, cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        if max(h,w) >1024:
            scale, img = resize2(img, 1024)
            scale, mask = resize2(mask, 1024)

        result = algorithm.forward(img, mask)

        cv2.imwrite(f"{DEST}/{fname}_out.jpg", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{DEST}/{fname}_in.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"{f} done")
