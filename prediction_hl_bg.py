# coding=utf-8
import datetime
import argparse
import logging
import pandas as pd
import os
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
graph1 = tf.Graph()
graph2 = tf.Graph()

graph_path_feicui = r"segmentation_background.pb"
graph_path_gaoguang = r"segmentation_highlight.pb"

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
    # gray_img = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    erode_img = cv2.erode(binary, kernel)
    fill_img = cv2.bitwise_and(image, image, mask=erode_img)
    return fill_img


def get_mask(image):
    # image = cv2.split(image)[0]
    image = np.array(image)
    LUT = np.zeros(256, dtype=np.uint8)
    for i in range(1, 20):
        LUT[i] = 200
    result = LUT[image]

    return result


def jadite_classify(image_name, path,label):
    image_path = os.path.join(path, image_name)
    org_image = cv2.imread(image_path)
    h, w, _ = org_image.shape

    resize_img = resize_image(org_image, 513, 513)
    # 3-模型预测
    image1 = model_pre1(resize_img)
    image11 = get_mask(image1)
    image12 = cv2.cvtColor(np.asarray(image11), cv2.COLOR_RGB2BGR)
    #背景去除图片
    image16 = pick_black(image12, resize_img)

    # 4-高光去除
    rm_hl_image = model_pre2(image16)

    rm_hl_mask = get_mask(rm_hl_image)

    #高光去除图片
    fill_hl_image = fill(rm_hl_mask, image16)
    save_path="/home/ubuntu/Image/gqf_DCL-master/dataset/fujian_guanyin_shui/no_lighlt_image/{0}".format(label)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cv2.imwrite("{0}/{1}".format(save_path,image_name),fill_hl_image)
    #高光去除原尺寸图片
    # orgin_hl = recovery_image(fill_hl_image, h, w)
    # cv2.imwrite("/home/ubuntu/Image/gqf_DCL-master/dataset/wushipai_class_zhong/add_no_light_image/{}".format(image_name),orgin_hl)




image_paths = r"/home/ubuntu/Image/gqf_DCL-master/dataset/fujian_guanyin_shui/source_image"

for label in os.listdir(image_paths):
    image_path=os.path.join(image_paths,label)
    for i in os.listdir(image_path):
        single_image_path=os.path.join(image_path,i)
        print(single_image_path)
        org_image = cv2.imread(single_image_path)
        if  org_image is not None:
            jadite_classify(i, image_path,label)


ssda