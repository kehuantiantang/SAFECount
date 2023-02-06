# coding=utf-8
# @Project  ：SAFECount 
# @FileName ：image_process.py
# @Author   ：SoberReflection
# @Revision : sober 
# @Date     ：2023/2/4 4:26 下午
import json
import multiprocessing
from multiprocessing.pool import ThreadPool

from tqdm import tqdm

from data.Chicken.gen_gt_density import points2density, apply_scoremap
import os.path as osp
import os

def show_image(img, title='', cmap = 'gray'):
    plt.figure()
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.show()

def reshape_points(points, a, b, c, d):
    tp_points = []
    for (x, y) in points:
        if x > d or y > b:
            continue
        else:
            x = x - c
            y = y - a
            if x < 0 or y < 0:
                continue
            else:
                tp_points.append((x, y))
    return tp_points

def save2file(target_path, filename, img, points, density, vis_heatmap = False):
    gt_density_map = osp.join(target_path, 'gt_density_map')
    frames = osp.join(target_path, 'frames')
    os.makedirs(gt_density_map, exist_ok=True)
    os.makedirs(frames, exist_ok=True)

    name = osp.splitext(filename)[0]
    json_content = {'filename':name + '.jpg', 'density': name + '.npy',
     'points':
         points.tolist()}
    with open(osp.join(frames, name+'.json'), 'w', encoding='utf-8') as f:
        json.dump(json_content, f, indent=4)

    np.save(osp.join(gt_density_map, name+'.npy'), density)

    cv2.imwrite(os.path.join(frames, name + '.jpg'), img)

    if vis_heatmap:
        min, max = density.min(), density.max()
        density = (density - min) / (max - min + 1e-8)
        mask = apply_scoremap(img, density)
        cv2.imwrite(os.path.join(gt_density_map, name + '_vis.jpg'), mask)



def operation(root, filename):
    img = cv2.imread(osp.join(root, filename))


    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (5,5))

    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    thresh = 255 - thresh
    kernel = np.ones((9,9),np.uint8)
    erode = cv2.erode(thresh, kernel, iterations = 1)



    minLineLength = 500
    maxLineGap = 10
    lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, 100, minLineLength, maxLineGap)

    rm_line = erode.copy()
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = line
        if abs(y1 - y2)^2 + abs(x1 - x2)^2 > 70:
            rm_line = cv2.line(rm_line, (x1, y1), (x2, y2), (0, 0, 0), 5)

    rm_line =cv2.blur(rm_line, (9,9))
    contours, hierarchy = cv2.findContours(rm_line, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    final_contours, points = [], []
    for i, contour in enumerate(contours):
        # ignore the small polygon
        if len(contour) < 10 or cv2.contourArea(contour) < 8*8:
            continue
        contour = np.array(contour)
        final_contours.append(contour)
        points.append([contour[:, :, 0].mean(), contour[:, :, 1].mean()])


    a, b, c, d = 300, 1000, 300, 2500
    img = img[a:b, c:d, :]

    points = reshape_points(points, a, b, c, d)

    points = np.array(points)

    cnt_gt = points.shape[0]
    density = points2density(points, max_scale=3.0, max_radius=15.0, image_size = img.shape[:2])

    if not cnt_gt == 0:
        cnt_cur = density.sum()
        density = density / cnt_cur * cnt_gt


    save2file(target_path, filename, img, points, density, True)

if __name__ == '__main__':
    # ostd method
    import numpy as np
    import cv2
    from matplotlib import pyplot as plt
    source_path, target_path = '/Volumes/datasets/chicken_count', '/Volumes/datasets/chicken_count_process'


    files = []
    for root, _, filenames in os.walk(source_path):
        for filename in filenames:
            if filename.endswith('jpg'):
                files.append((root, filename))

    bar = tqdm(total=len(files), desc='total')
    update = lambda *args: bar.update()

    pool = ThreadPool(10)


    for root, filename in sorted(files):
        kwds = {'root':root, 'filename':filename}
        # video2frame(**kwds)
        pool.apply_async(operation, kwds=kwds, callback=update)

    pool.close()
    pool.join()

