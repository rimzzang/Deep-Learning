import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import copy

from os.path import join
from pycocotools.coco import COCO
from dataset_prepare import CocoMetadata#, CocoPose

#BASE = "/home/bglee86/work/c_pose_p/"
IMAGE_PATH = "/home/leeserim/Downloads/val2017"
#BASE_PATH = "/home/bglee86/work/c_pose_p/ai_challenger"
ANNO_PATH = "/home/leeserim/Downloads/annotations"

def display_original_image(img_path, annotations):
    img_str = open(img_path, "rb").read()
    if not img_str:
        print("image not read, path=%s" % img_path)
    nparr = np.fromstring(img_str, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    inp = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    plt.imshow(inp)
    plt.axis('off')
    plt.show()

def pose_to_img(meta_l):
    return meta_l.img.astype(np.flot32), meta_l.get_heatmap(target_size=(meta_l.width, meta_l.height)).astype(np.float32)


def _parse_function(imgId, is_train, ann):

    global TRAIN_ANNO
    global VALID_ANNO

    anno = ann
    img_meta = anno.loadImgs([imgId])[0]
    anno_ids = anno.getAnnIds(imgIds=imgId)
    img_anno = anno.loadAnns(anno_ids)

    idx = img_meta['id']
    img_path = join(IMAGE_PATH, img_meta['file_name'])
    img_meta_data = CocoMetadata(idx, img_path, img_meta, img_anno, sigma=6.0)
    display_original_image(img_path, img_anno)
    heatmap = img_meta_data.get_heatmap((img_meta_data.height,img_meta_data.width))
    leng = len(img_meta_data.joint_list[0])
    joint = img_meta_data.joint_list[0]
    plt.figure(1, figsize=(30,30))
    n_columns = 5
    n_rows = math.ceil(leng / n_columns)
    b, g, r = cv2.split(img_meta_data.img)
    img_resize = cv2.merge((r,g,b))

    i=5
    for i in range(leng):
#        img_resize = copy.copy(img_meta_data.img)
        if i>4:
            #plt.subplot(n_rows, n_columns, i + 1)
            #plt.title('Joint Point' + str(i))
            img_resize = cv2.circle(img_resize, joint[i], 5, (0, 255, 0), cv2.FILLED)#CV2.filled #채워진 점
            plt.imshow(img_resize)

    #추가 coco api사용해서 점을 잇는 함수
    anno.showAnns(img_anno)

    set_p = [point for point in img_meta_data.joint_list[0]]
    head_point = cv2.circle(img_resize,
                            ((set_p[2][0]+set_p[1][0])//2,
                             (set_p[2][1]+set_p[1][1])*3//2-set_p[0][1]*2)
                            , 5, (255, 0, 0), cv2.FILLED)
    neck_point = cv2.circle(img_resize,
                            ((set_p[6][0]+set_p[5][0])//2,
                             (set_p[6][1]+set_p[5][1])//3+ set_p[0][1]//3)
                            , 5, (255, 0, 0), cv2.FILLED)
    plt.imshow(head_point)
    plt.imshow(neck_point)
    plt.show()


def display_image():
    #ANNO = COCO(
    #    join(BASE_PATH, "ai_challenger_valid.json")
    #)
    ANNO = COCO(
        join(ANNO_PATH, "person_keypoints_val2017.json")
    )
    train_imgIds = ANNO.getImgIds()
    _parse_function(train_imgIds[11], False, ANNO)

    #CocoPose.display_image(img, heat, pred_heat=heat, as_numpy=False)


#    img_meta_data = CocoMetadata(idx, img_path, img_meta, img_anno, sigma=6.0)

#    target_size = (img_meta['width'], img_meta['height'])
#    img_meta_data.width, img_meta_data.height = target_size

#    return pose_to_img(img_meta_data)

if __name__ == '__main__':
    display_image()
