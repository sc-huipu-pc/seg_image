# coding=utf-8
import os
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
import cv2
import numpy as np
import pandas as pd

graph1 = tf.Graph()
graph2 = tf.Graph()

graph_path_feicui = r"./segmentation_background.pb"
graph_path_gaoguang = r"./segmentation_highlight.pb"

with tf.gfile.FastGFile(graph_path_feicui, 'rb') as f1:
    graph_def1 = tf.GraphDef()
    graph_def1.ParseFromString(f1.read())
if graph_def1 is None:
    raise RuntimeError('模型不存在.')
with graph1.as_default():
    tf.import_graph_def(graph_def1, name="")

with tf.gfile.FastGFile(graph_path_gaoguang, "rb")as f2:
    graph_def2 = tf.GraphDef()
    graph_def2.ParseFromString(f2.read())
if graph_def2 is None:
    raise RuntimeError("模型不存在")
with graph2.as_default():
    tf.import_graph_def(graph_def2, name="")

INPUT_TENSOR_NAME = 'ImageTensor:0'
OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'

sess1 = tf.Session(graph=graph1)
sess2 = tf.Session(graph=graph2)

"""
固定参数初始化
color_name_list：颜色字段
draw_contours_list：画图字段
h颜色通道的取值范围：h_min_list ， h_max_list
s取值通道的范围：s_min_list，s_max_list
b取值通道的范围：b_min_list，b_max_list
画图颜色：RBG三通道
    draw_r_list
    draw_g_list
    draw_b_list
"""

# color_name_list = ["红色", "橘色", "黄色", "黄绿色", "绿色", "绿蓝色", "蓝色", "蓝紫色", "紫蓝色", "紫红色", "白色", "灰色", "黑色"]
# color_name_list = ["红色", "橘色", "黄色", "黄绿色", "绿色", "蓝绿色", "蓝色", "紫蓝色", "蓝紫色", "粉紫色", "白色", "灰色", "黑色"]
# draw_contours_list = ["绿蓝色", "蓝色", "蓝紫色", "紫蓝色", "紫红色", "红色", "橘色", "黄色", "黄绿色", "绿色", "黑色", "白色", "白色"]
# h_min_list = [0, 9, 18, 35, 60, 79, 93, 108, 115, 142, 0, 0, 1]
# h_max_list = [8, 17, 34, 59, 78, 92, 107, 114, 141, 169, 180, 180, 180]
# s_min_list = [60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 0, 0, 1]
# s_max_list = [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 60, 60, 255]
# b_min_list = [46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 153, 46, 1]
# b_max_list = [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 152, 46]
# draw_r_list = [0, 0, 0, 140, 255, 255, 255, 255, 190, 30, 0, 255, 255]
# draw_g_list = [255, 150, 20, 0, 0, 0, 80, 170, 255, 255, 0, 255, 255]
# draw_b_list = [255, 255, 255, 255, 180, 50, 0, 0, 0, 0, 0, 255, 255]
color_name_list = ["红色", "橘色", "黄色", "黄绿色", "绿色", "蓝绿色", "蓝色", "紫蓝色", "蓝紫色", "粉紫色", "白色", "黑色"]
draw_contours_list = ["绿蓝色", "蓝色", "蓝紫色", "紫蓝色", "紫红色", "红色", "橘色", "黄色", "黄绿色", "绿色", "黑色", "白色"]
h_min_list = [0, 9, 18, 35, 60, 79, 93, 108, 115, 142, 0, 1]
h_max_list = [8, 17, 34, 59, 78, 92, 107, 114, 141, 169, 180, 180]
s_min_list = [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 0, 1]
s_max_list = [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 12, 255]
b_min_list = [25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 1]
b_max_list = [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 25]
draw_r_list = [0, 0, 0, 140, 255, 255, 255, 255, 190, 30, 0, 255]
draw_g_list = [255, 150, 20, 0, 0, 0, 80, 170, 255, 255, 0, 255]
draw_b_list = [255, 255, 255, 255, 180, 50, 0, 0, 0, 0, 0, 255]


def new_color_area_detection(hsv_img, pixel_total_num, bgr_img, kernel, color_dict, color_name, draw_color_name,
                             draw_r, draw_g, draw_b, h_min, h_max, s_min, s_max, b_min, b_max):
    if color_name == "红色":
        red_img_low = cv2.inRange(hsv_img, (h_min, s_min, b_min), (h_max, s_max, b_max))
        red_img_high = cv2.inRange(hsv_img, (170, s_min, b_min), (180, s_max, b_max))
        color_area = cv2.add(red_img_low, red_img_high)
    else:
        color_area = cv2.inRange(hsv_img, (h_min, s_min, b_min), (h_max, s_max, b_max))

    color_pixel_ratio = int(len(color_area[color_area > 0]) / pixel_total_num * 100)

    if color_pixel_ratio < 5:
        color_area = None
        return color_dict, color_area, bgr_img
    else:
        color_area = cv2.erode(color_area, kernel)
        contours, hierarchy = cv2.findContours(image=color_area, mode=cv2.RETR_EXTERNAL,
                                               method=cv2.CHAIN_APPROX_SIMPLE)
        contours_list = []
        contours_max = []
        if not contours:
            color_dict["color_area"][color_name] = {"占比": str(color_pixel_ratio) + "%"}
            color_dict["color_area"][color_name]["边界颜色"] = draw_color_name
            color_dict["color_area"][color_name]["RGB数值"] = (draw_r, draw_g, draw_b)
            cv2.drawContours(bgr_img, contours_list, -1, (draw_b, draw_g, draw_r), 2)
            return color_dict, color_area, bgr_img
        else:
            for c in contours:
                contours_max.append(len(c))
                if cv2.contourArea(c) > 3000:
                    contours_list.append(c)

            if not contours_list:
                contours_list.append(contours[contours_max.index(max(contours_max))])
            color_dict["color_area"][color_name] = {"占比": str(color_pixel_ratio) + "%"}
            color_dict["color_area"][color_name]["边界颜色"] = draw_color_name
            color_dict["color_area"][color_name]["RGB数值"] = (draw_r, draw_g, draw_b)
            cv2.drawContours(bgr_img, contours_list, -1, (draw_b, draw_g, draw_r), 2)

    return color_dict, color_area, bgr_img


def area_grade(color_name, color_area, org_img, color_dict):
    """
    实现评级
    1-拿到颜色区域
    2-分析hsv下的饱和度，明度，均匀度等级

    :param color_name: 颜色名字
    :param color_area: 颜色区域——图片
    :param org_img: 原始图片
    :param color_dict: 字典信息
    :return: 返回评级字典
    """
    if color_area is None:
        return color_dict

    # 根据掩模图，拿到颜色区域原图，并转hsv
    area_img = cv2.bitwise_and(org_img, org_img, mask=color_area)
    area_hsv_img = cv2.cvtColor(area_img, cv2.COLOR_BGR2HSV)

    # 统计频域上的最大数值数值，以及集中区域的阈值，s通道，b通道。
    s_hist = cv2.calcHist([area_hsv_img], [1], None, [256], [1, 255])
    s_grade = np.where(s_hist == np.max(s_hist))[0][0]
    b_hist = cv2.calcHist([area_hsv_img], [2], None, [256], [1, 255])
    b_grade = np.where(b_hist == np.max(b_hist))[0][0]
    s_thresh = int(np.max(s_hist) * 0.05)
    b_thresh = int(np.max(b_hist) * 0.05)

    # 拿到集中的差值，90%集中度的范围。
    s_differ = np.max(np.where(s_hist >= s_thresh)[0]) - np.min(np.where(s_hist >= s_thresh)[0])
    b_differ = np.max(np.where(b_hist >= b_thresh)[0]) - np.min(np.where(b_hist >= b_thresh)[0])
    total_differ = s_differ + b_differ

    # color_name_list = ["红色", "橘色", "黄色", "黄绿色", "绿色", "蓝绿色", "蓝色", "紫蓝色", "蓝紫色", "粉紫色", "白色", "灰色", "黑色"]
    # 绿色，黄绿色，蓝绿色
    if color_name == color_name_list[4] or color_name == color_name_list[3] or color_name == color_name_list[5]:

        # 90-100
        if s_grade >= 230:
            color_dict["color_area"][color_name]["饱和度"] = "极浓"
        # 80-90
        elif 204 <= s_grade < 230:
            color_dict["color_area"][color_name]["饱和度"] = "浓"
        # 50-80
        elif 128 <= s_grade < 204:
            color_dict["color_area"][color_name]["饱和度"] = "适中"
        else:
            color_dict["color_area"][color_name]["饱和度"] = "浅"

        # 50-100 不灰
        if 128 <= b_grade:
            color_dict["color_area"][color_name]["明度"] = "不灰"
        # 30-50 轻微灰
        elif 76 <= b_grade < 128:
            color_dict["color_area"][color_name]["明度"] = "轻微灰"
        # 20-30 灰
        elif 51 <= b_grade < 76:
            color_dict["color_area"][color_name]["明度"] = "灰"
        else:
            color_dict["color_area"][color_name]["明度"] = "极灰"

    # color_name_list = ["红色", "橘色", "黄色", "黄绿色", "绿色", "蓝绿色", "蓝色", "紫蓝色", "蓝紫色", "粉紫色", "白色", "灰色", "黑色"]
    # 红色、橘色
    elif color_name == color_name_list[0] or color_name == color_name_list[1]:

        # 80-100
        if s_grade >= 204:
            color_dict["color_area"][color_name]["饱和度"] = "极浓"
        # 60-80
        elif 178 <= s_grade < 204:
            color_dict["color_area"][color_name]["饱和度"] = "浓"
        # 50-60
        elif 128 <= s_grade < 178:
            color_dict["color_area"][color_name]["饱和度"] = "适中"
        else:
            color_dict["color_area"][color_name]["饱和度"] = "浅"

        # 80-100 不灰
        if b_grade >= 204:
            color_dict["color_area"][color_name]["明度"] = "不灰"
        # 60-80 轻微灰
        elif 178 <= b_grade < 204:
            color_dict["color_area"][color_name]["明度"] = "轻微灰"
        # 40-60 灰
        elif 102 <= b_grade < 178:
            color_dict["color_area"][color_name]["明度"] = "灰"
        else:
            color_dict["color_area"][color_name]["明度"] = "极灰"

    # color_name_list = ["红色", "橘色", "黄色", "黄绿色", "绿色", "蓝绿色", "蓝色", "紫蓝色", "蓝紫色", "粉紫色", "白色", "灰色", "黑色"]
    # 黄色
    elif color_name == color_name_list[2]:

        # 90-100
        if s_grade >= 230:
            color_dict["color_area"][color_name]["饱和度"] = "极浓"
        # 70-90
        elif 178 <= s_grade < 230:
            color_dict["color_area"][color_name]["饱和度"] = "浓"
        # 50-70
        elif 128 <= s_grade < 178:
            color_dict["color_area"][color_name]["饱和度"] = "适中"
        else:
            color_dict["color_area"][color_name]["饱和度"] = "浅"

        # 80-100 不会
        if b_grade >= 204:
            color_dict["color_area"][color_name]["明度"] = "不灰"
        # 60-80 轻微灰
        elif 178 <= b_grade < 204:
            color_dict["color_area"][color_name]["明度"] = "轻微灰"
        # 40-60 灰
        elif 102 <= b_grade < 178:
            color_dict["color_area"][color_name]["明度"] = "灰"
        else:
            color_dict["color_area"][color_name]["明度"] = "极灰"

    # 蓝色、紫蓝色
    elif color_name == color_name_list[6] or color_name == color_name_list[7]:

        # 90-100
        if s_grade >= 230:
            color_dict["color_area"][color_name]["饱和度"] = "极浓"
        # 80-90
        elif 204 <= s_grade < 230:
            color_dict["color_area"][color_name]["饱和度"] = "浓"
        # 50-80
        elif 128 <= s_grade < 204:
            color_dict["color_area"][color_name]["饱和度"] = "适中"
        else:
            color_dict["color_area"][color_name]["饱和度"] = "浅"

        # 50-100 不灰
        if 128 <= b_grade:
            color_dict["color_area"][color_name]["明度"] = "不灰"
        # 30-50 轻微灰
        elif 76 <= b_grade < 128:
            color_dict["color_area"][color_name]["明度"] = "轻微灰"
        # 20-30 灰
        elif 51 <= b_grade < 76:
            color_dict["color_area"][color_name]["明度"] = "灰"
        else:
            color_dict["color_area"][color_name]["明度"] = "极灰"

    # color_name_list = ["红色", "橘色", "黄色", "黄绿色", "绿色", "蓝绿色", "蓝色", "紫蓝色", "蓝紫色", "粉紫色", "白色", "灰色", "黑色"]
    # 蓝紫色、粉紫色
    elif color_name == color_name_list[8] or color_name == color_name_list[9]:

        # 60-100 极浓
        if s_grade >= 153:
            color_dict["color_area"][color_name]["饱和度"] = "极浓"
        # 40-60 浓
        elif 102 <= s_grade < 153:
            color_dict["color_area"][color_name]["饱和度"] = "浓"
        # 30-40 适中
        elif 76 <= s_grade < 102:
            color_dict["color_area"][color_name]["饱和度"] = "适中"
        else:
            color_dict["color_area"][color_name]["饱和度"] = "浅"

        # 80-100 不灰
        if b_grade >= 204:
            color_dict["color_area"][color_name]["明度"] = "不灰"
        # 60-80 轻微灰
        elif 178 <= b_grade < 204:
            color_dict["color_area"][color_name]["明度"] = "轻微灰"
        # 40-60 灰
        elif 102 <= b_grade < 178:
            color_dict["color_area"][color_name]["明度"] = "灰"
        else:
            color_dict["color_area"][color_name]["明度"] = "极灰"

    # 白色
    elif color_name == color_name_list[10]:

        # # 90-100 不灰
        # if b_grade >= 204:
        #     color_dict["color_area"][color_name]["明度"] = "不灰"
        # # 80-90 轻微灰
        # elif 178 <= b_grade < 204:
        #     color_dict["color_area"][color_name]["明度"] = "轻微灰"
        # # 70-80 灰
        # elif 102 <= b_grade < 178:
        #     color_dict["color_area"][color_name]["明度"] = "灰"
        # else:
        #     color_dict["color_area"][color_name]["明度"] = "极灰"

        # 70-100 不灰
        if b_grade >= 178:
            color_dict["color_area"][color_name]["明度"] = "不灰"
        # 50-70 轻微灰
        elif 127 <= b_grade < 178:
            color_dict["color_area"][color_name]["明度"] = "轻微灰"
        # 30-50 灰
        elif 76 <= b_grade < 178:
            color_dict["color_area"][color_name]["明度"] = "灰"
        else:
            color_dict["color_area"][color_name]["明度"] = "极灰"

    # # 灰色
    # elif color_name == color_name_list[11]:
    #
    #     # 灰色低于23, 换算为60
    #     if 45 <= s_grade < 60:
    #         color_dict["color_area"][color_name]["饱和度"] = "极浓"
    #     # 80-90 轻微灰
    #     elif 30 <= s_grade < 45:
    #         color_dict["color_area"][color_name]["饱和度"] = "浓"
    #     # 70-80 灰
    #     elif 15 <= s_grade < 30:
    #         color_dict["color_area"][color_name]["饱和度"] = "适中"
    #     else:
    #         color_dict["color_area"][color_name]["饱和度"] = "浅"
    #
    #     # 低于60 不灰 15-60
    #     if 127 <= b_grade < 153:
    #         color_dict["color_area"][color_name]["明度"] = "不灰"
    #     # 轻微灰
    #     elif 101 <= b_grade < 127:
    #         color_dict["color_area"][color_name]["明度"] = "轻微灰"
    #     # 灰
    #     elif 75 <= b_grade < 101:
    #         color_dict["color_area"][color_name]["明度"] = "灰"
    #     else:
    #         color_dict["color_area"][color_name]["明度"] = "极灰"
    else:

        color_dict["color_area"][color_name]["饱和度"] = "无"
        color_dict["color_area"][color_name]["明度"] = "无"

    # color_name_list = ["红色", "橘色", "黄色", "黄绿色", "绿色", "蓝绿色", "蓝色", "紫蓝色", "蓝紫色", "粉紫色", "白色", "灰色", "黑色"]
    # "红色", "橘色", "黄色", "黄绿色", "绿色", "蓝绿色", "蓝色", "紫蓝色", "蓝紫色", "粉紫色"
    if color_name in color_name_list[:10]:
        if total_differ <= 136:
            color_dict["color_area"][color_name]["均匀性"] = "均匀"
        elif 136 < total_differ <= 202:
            color_dict["color_area"][color_name]["均匀性"] = "较均匀"
        else:
            color_dict["color_area"][color_name]["均匀性"] = "不均匀"

    # 白色
    elif color_name == color_name_list[10]:
        if total_differ <= 54:
            color_dict["color_area"][color_name]["均匀性"] = "均匀"
        elif 54 < total_differ <= 98:
            color_dict["color_area"][color_name]["均匀性"] = "较均匀"
        else:
            color_dict["color_area"][color_name]["均匀性"] = "不均匀"

    # # 灰色
    # elif color_name == color_name_list[11]:
    #     if total_differ <= 59:
    #         color_dict["color_area"][color_name]["均匀性"] = "均匀"
    #     elif 59 < total_differ <= 118:
    #         color_dict["color_area"][color_name]["均匀性"] = "较均匀"
    #     else:
    #         color_dict["color_area"][color_name]["均匀性"] = "不均匀"
    else:
        color_dict["color_area"][color_name]["均匀性"] = "无"

    return color_dict


def main(org_img, write_path, color_dict, pixel_total_num, image_name):
    # 备份图片
    bgr_img = org_img.copy()

    # 转HSV颜色空间
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    # 获取核函数
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # 循环查找对应颜色区域
    for i, color_name in enumerate(color_name_list):
        # 颜色区域检测
        color_dict, color_area, bgr_img = new_color_area_detection(hsv_img, pixel_total_num, bgr_img, kernel,
                                                                   color_dict, color_name, draw_contours_list[i],
                                                                   draw_r_list[i], draw_g_list[i], draw_b_list[i],
                                                                   h_min_list[i], h_max_list[i],
                                                                   s_min_list[i], s_max_list[i],
                                                                   b_min_list[i], b_max_list[i])
        if color_area is None:
            continue
        color_area_write_name = "color_%s_seg_" % i + image_name
        color_dict["color_area"][color_name]["write_name"] = color_area_write_name
        color_area_write_path = os.path.join(write_path, color_area_write_name)
        cv2.imwrite(color_area_write_path, color_area)
        # 颜色区域评级
        color_dict = area_grade(color_name, color_area, org_img, color_dict)
    write_name = "seg_" + image_name
    color_dict["image_name"] = write_name
    color_dict["valid"] = 1
    write_path_total = os.path.join(write_path, write_name)
    # 写入本地最新带有轮廓的图片
    cv2.imwrite(write_path_total, bgr_img)
    return color_dict


def resize_image(image, height, width):
    # 图片尺寸缩放
    top, bottom, left, right = (0, 0, 0, 0)
    h, w, _ = image.shape
    longest_edge = max(h, w)
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass
    BLACK = [0, 0, 0]
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)
    resize_image = cv2.resize(constant, (height, width))
    return resize_image


def recovery_image(image, h, w):
    top, bottom, left, right = (0, 0, 0, 0)
    min_edge = min(h, w)
    if w > min_edge:
        dw = w - min_edge
        left = dw // 2
        right = dw - left
    elif h > min_edge:
        dh = h - min_edge
        top = dh // 2
        bottom = dh - top
    BLACK = [0, 0, 0]
    resize_image = cv2.resize(image, (min_edge, min_edge))
    constant = cv2.copyMakeBorder(resize_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)
    return constant


def model_pre1(img):
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0).astype(np.uint8)
    result = sess1.run(OUTPUT_TENSOR_NAME, feed_dict={INPUT_TENSOR_NAME: img})
    image = result.transpose((1, 2, 0))
    return image


def model_pre2(img):
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0).astype(np.uint8)
    result = sess2.run(
        OUTPUT_TENSOR_NAME,
        feed_dict={INPUT_TENSOR_NAME: img})
    image = result.transpose((1, 2, 0))
    return image


def pick_black(mask, image):
    gray_img = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    erode_img = cv2.erode(binary, kernel, iterations=4)
    dilate_img = cv2.dilate(erode_img, kernel, iterations=8)
    erode_img = cv2.erode(dilate_img, kernel, iterations=5)
    fill_img = cv2.bitwise_and(image, image, mask=erode_img)
    return fill_img


def fill(gray_img, image):
    ret, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    erode_img = cv2.erode(binary, kernel)
    fill_img = cv2.bitwise_and(image, image, mask=erode_img)
    return fill_img


def get_mask(image):
    image = np.array(image)
    LUT = np.zeros(256, dtype=np.uint8)
    for i in range(1, 20):
        LUT[i] = 200
    result = LUT[image]

    return result


def handle(image_name, org_path, write_path):
    color_dict = {}
    color_dict["image_name"] = None
    color_dict["valid"] = 0
    color_dict["message"] = {"statu": 0, "content": None}
    color_dict["color_area"] = {}
    image_path = os.path.join(org_path, image_name)
    if not os.path.isfile(image_path):
        color_dict["message"]["statu"] = 1
        color_dict["message"]["content"] = "图片路径有问题，不是文件"
        return color_dict
    org_image = cv2.imread(image_path)
    if org_image is None:
        color_dict["message"]["statu"] = 1
        color_dict["message"]["content"] = "非图片数据，读取失败"
        return color_dict
    h, w, _ = org_image.shape

    resize_img = resize_image(org_image, 513, 513)

    # 3-模型预测
    image1 = model_pre1(resize_img)
    image11 = get_mask(image1)
    image12 = cv2.cvtColor(np.asarray(image11), cv2.COLOR_RGB2BGR)

    image16 = pick_black(image12, resize_img)

    # 4-高光去除
    rm_hl_image = model_pre2(image16)
    rm_hl_mask = get_mask(rm_hl_image)

    fill_hl_image = fill(rm_hl_mask, image16)

    orgin_hl = recovery_image(fill_hl_image, h, w)

    vaild_pixel = np.logical_not(np.logical_and(np.logical_and(orgin_hl[:, :, 0] < 1, orgin_hl[:, :, 1] < 1),
                                                orgin_hl[:, :, 2] < 1))
    pixel_total_num = len(orgin_hl[vaild_pixel])
    vaild_pixel_ratio = int(len(orgin_hl[vaild_pixel]) / (h * w) * 100)

    if vaild_pixel_ratio <= 1:
        color_dict["message"]["statu"] = 1
        color_dict["message"]["content"] = "分割图像和去高光效果不佳，导致图片损失严重"
        return color_dict
    color_dict = main(orgin_hl, write_path, color_dict, pixel_total_num, image_name)
    return color_dict

if __name__ == '__main__':

    path = r"/home/ubuntu/Image/seg_image/test_image/yijian"
    write_path = r"./color_seg_data"
    total_list = []
    for image_name in os.listdir(path):

        color_dict = handle(image_name, path, write_path)
        total_list.append(color_dict)
        print(color_dict)
    data = {"total_data": total_list}
    df = pd.DataFrame(data)
    df.to_csv(r"/home/ubuntu/Image/seg_image/total_data.csv")
