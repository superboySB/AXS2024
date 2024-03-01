import cv2
from matplotlib import pyplot as plt

def vis_image_and_depth(rgb_image, depth_image):
    # 读取深度图像（这里假设深度图是单通道灰度图）

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            depth_value = depth_image[y, x]  # 获取鼠标点击处的深度值
            print(f"Depth value at ({x}, {y}): {depth_value} mm")

    cv2.imshow('Depth Image', rgb_image)
    cv2.setMouseCallback('Depth Image', on_mouse)

    plt.title("Depth Image")
    plt.imshow(depth_image, cmap='jet')  # 使用jet colormap进行可视化
    plt.colorbar(label='Depth (mm)')
    plt.show()
    while True:
        cv2.waitKey()
        cv2.destroyAllWindows()