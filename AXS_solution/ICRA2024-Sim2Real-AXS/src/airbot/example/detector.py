# %%
from airbot.backend import Arm, Camera, Base, Gripper
import os
import numpy as np
from airbot.lm import Detector
from PIL import Image
import time
import cv2
from airbot.example.utils.draw import draw_bbox, obb2poly

os.environ['LM_CONFIG'] = "/root/Workspace/AXS_baseline/ICRA2024-Sim2Real-AXS/local.yaml"
os.environ['CKPT_DIR'] = '/root/Workspace/AXS_baseline/ckpt'

inspect_pose = (np.array([0.3379587248352459, 0.013964714471332657, 0.11577598561175462]),
                np.array([-0.002536924036797276, 0.15437602025710206, 0.02049498220919601, 0.9877963171070517]))

arm = Arm(backend='ros')
arm.move_end_to_pose(*inspect_pose)

camera = Camera(backend='ros')
detector = Detector(model='grounding-dino')

# %%

try:
    # Infinite loop to display images
    while True:
        image = camera.get_rgb()

        result = detector.infer(image, 'cup')
        image = draw_bbox(image, obb2poly(result['bbox'].unsqueeze(0)).astype(int))

        cv2.imshow('RGB', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.1)

except KeyboardInterrupt:
    print("Exiting due to user interruption.")

finally:
    cv2.destroyAllWindows()
