import numpy as np
import cv2
import math

class CocoMetadata:
    #for c_pose
    #__coco_parts = 14
    __coco_parts = 19

    def __init__(self, idx, img_path, img_meta, annotations, sigma):
        self.idx = idx
        self.img = self.read_image(img_path)
        self.sigma = sigma

        self.height = int(img_meta['height'])
        self.width = int(img_meta['width'])

        self.joint_list = []
        for ann in annotations:
            if ann.get('num_keypoints', 0) == 0:
                continue

            kp = np.array(ann['keypoints'])
            xs = kp[0::3]
            ys = kp[1::3]
            vs = kp[2::3]

            self.joint_list.append([(x,y) if v >=1 else (-1000, -1000) for x, y, v in zip(xs, ys, vs)])

        joint_list = []

        # joint point order change for depplearning network model
        transform = list(zip(
            [1, 2, 4, 6, 8, 3, 5, 7, 10, 12, 14, 9, 11, 13],
            [1, 2, 4, 6, 8, 3, 5, 7, 10, 12, 14, 9, 11, 13]
        ))

        for prev_joint in joint_list:
            new_joint = []
            for idx1, idx2 in transform:
                j1 = prev_joint[idx1 - 1]
                j2 = prev_joint[idx2 - 1]

                if j1[0] <= 0 or j1[1] <= 0 or j2[0] <= 0 or j2[1] <= 0:
                    new_joint.append((-1000, -1000))
                else:
                    new_joint.append((int((j1[0] + j2[0]) / 2), (int(j1[1] + j2[1] /2))))

            joint_list.append(new_joint)

    def get_heatmap(self, target_size):
        heatmap = np.zeros((CocoMetadata.__coco_parts, target_size[0], target_size[1]), dtype=np.float32)

        for joints in self.joint_list:
            for idx, point in enumerate(joints):
                if point[0] < 0 or point[1] < 0:
                    continue
                CocoMetadata.put_heatmap(heatmap, idx, point, self.sigma)

        heatmap = heatmap.transpose((1, 2, 0))



        if target_size:
            heatmap = cv2.resize(heatmap, target_size, interpolation=cv2.INTER_AREA)

        return heatmap.astype(np.float16)

    def put_heatmap(heatmap, plane_idx, center, sigma):
        center_x, center_y = center
        _, height, width = heatmap.shape[:3]

        th = 1.6052
        delta = math.sqrt(th * 2)

        x0 = int(max(0, center_x - delta * sigma))
        y0 = int(max(0, center_y - delta * sigma))

        x1 = int(min(width, center_x + delta * sigma))
        y1 = int(min(height, center_y + delta * sigma))
        #print("x0 : %d, y0 : %d, x1 : %d, y1 : %d" %(x0, y0, x1, y1))
        # gaussian filter
        for y in range(y0, y1):
            for x in range(x0, x1):
                d = (x - center_x) ** 2 + (y - center_y) ** 2
                exp = d / 2.0 / sigma / sigma
                if exp > th:
                    continue
                heatmap[plane_idx][y][x] = max(heatmap[plane_idx][y][x], math.exp(-exp))
                heatmap[plane_idx][y][x] = min(heatmap[plane_idx][y][x], 1.0)

    def read_image(self, img_path):
        img_str = open(img_path, "rb").read()
        if not img_str:
            print("image not read, path=%s" % img_path)
        nparr = np.fromstring(img_str, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
