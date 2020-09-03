import argparse
import shutil
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
# import cv2
import numpy as np
import torch.nn as nn
import torch
import pandas as pd
import torch.backends.cudnn as cudnn
from torchvision import transforms
from config1 import LoadConfig
from models.LoadModel import MainModel
import os
from PIL import Image,ImageFont,ImageDraw
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
np.set_printoptions(suppress=True)
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,precision_score,recall_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import auc

from sklearn.metrics import  classification_report

INPUT_TENSOR_NAME = 'ImageTensor:0'
OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'

params_path_shui=r"/home/ubuntu/Image/gqf_DCL-master/net_model/_81211_fujian_merge_data_zhong/weights_29_347_0.7623_0.9446_0.9886.pth"
# image_path=r"/home/ubuntu/Image/gqf_DCL-master/dataset/merge_data_shui/data/test"
image_path=r"/home/ubuntu/Image/gqf_DCL-master/dataset/fujian_danmian_zhong_new/data/test"
# source_image_path="/home/ubuntu/Image/gqf_DCL-master/dataset/fujian_shuozhuo_shui/source_image"
check_image=r"/home/ubuntu/Image/gqf_DCL-master/dataset/fujian_danmian_zhong_new/source_image"

if __name__ == '__main__':

    Config = LoadConfig()
    device = torch.device("cuda")
    Config.cls_2 = False
    Config.cls_2xmul = True
    # Config.numcls = 5
    Config.numcls =8
    cudnn.benchmark = True
    model = MainModel(Config)
    # model.cuda()
    model_dict = model.state_dict()
    pretrained_dict = torch.load(params_path_shui, map_location=device)
    pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(pretrained_dict)
    model=model.to(device)
    model.eval()

    trans = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    count=0
    num=0
    y_pre=[]
    y_label=[]
    data_lists=[]
    dict_light={}

    with torch.no_grad():
        for j in os.listdir(image_path):
            # print("j",j)
            for i in os.listdir(os.path.join(image_path,j)):
                image = Image.open(os.path.join(image_path,j,i))
                #
                image=image.resize((448,448),Image.ANTIALIAS)

                image1 = trans(image)
                image2 = image1.unsqueeze(0)
                image2=image2.to(device)

                out = model(image2)

                pre = out[0]
                value, prediction = torch.max(pre, 1)
                #
                # pre = np.squeeze(pre)
                pre_value=prediction.item()

                # pre_label_dict = {0: "极透光", 1: "亚透光", 2: "半透光", 3: "微透光",4:"不透光"}
                # pre_label_dict = {0: "特浓", 1: "浓", 2: "适中", 3: "浅"}
                pre_label_dict = {0: "玻璃种", 1: "高冰种", 2: "冰种", 3: "细糯种", 4: "糯种",5:"细豆种",6:"豆种",7:"冰糯种"}
                # pre_label_dict = {0: "玻璃种", 1: "高冰种", 2: "冰种", 3: "细糯种", 4: "糯种",5:"冰糯种"}
                # pre_label_dict = {0: "不灰", 1: "轻微灰", 2: "灰", 3: "极灰"}
                # really_label_dict= {128: "特浓", 129: "浓", 130: "适中", 131: "浅"}
                really_label_dict= {6: "玻璃种", 7: "高冰种", 8: "冰种", 9: "细糯种",10:"糯种",11:"细豆种",12:"豆种",51:"冰糯种"}
                # really_label_dict= {6: "玻璃种", 7: "高冰种", 8: "冰种", 9: "细糯种",10:"糯种",51:"冰糯种"}
                # really_label_dict= {132: "不灰", 133: "轻微灰", 134: "灰", 135: "极灰"}
                # really_label_dict={13:"极透光",14:"亚透光",15:"半透光",16:"微透光",17:"不透光"}
                num += 1
                print("num:",num,"预测值为：",pre_label_dict[pre_value],"标签为：",really_label_dict[int(j)])
                if pre_label_dict[pre_value]==really_label_dict[int(j)]:

                    count+=1
                    print(count / num)

                # 水####
                # y_pre.append(pre_label_dict[pre_value])
                # y_label.append(really_label_dict[int(j)])

                # if really_label_dict[int(j)]=="微透光":
                #     dict_light[i]=[pre_label_dict[pre_value]]

                # if pre_label_dict[pre_value]=="半透光":
                #     diretion_path="/home/ubuntu/Image/gqf_DCL-master/dataset/fujian_shuozhuo_shui"
                #     save_path="{0}/".format(diretion_path)+"pre_{0}/".format(pre_label_dict[pre_value])+"true_{0}".format(really_label_dict[int(j)])
                #     if not os.path.exists(save_path):
                #         os.makedirs(save_path)
                #     if i in os.listdir(os.path.join(source_image_path,str(j))):
                #         shutil.copy(os.path.join(source_image_path,str(j),i),save_path)
                #
                # elif pre_label_dict[pre_value]=="微透光":
                #     diretion_path="/home/ubuntu/Image/gqf_DCL-master/dataset/fujian_shuozhuo_shui"
                #     save_path="{0}/".format(diretion_path)+"pre_{0}/".format(pre_label_dict[pre_value])+"true_{0}".format(really_label_dict[int(j)])
                #     if not os.path.exists(save_path):
                #         os.makedirs(save_path)
                #     if i in os.listdir(os.path.join(source_image_path,str(j))):
                #         shutil.copy(os.path.join(source_image_path,str(j),i),save_path)
                #明暗度###
                # y_pre.append(pre_label_dict[pre_value])
                # y_label.append(really_label_dict[int(j)])


                #种####
                # trans_dict={0:0,1:1,2:2,7:3,3:4,4:5,5:6,6:7}
                trans_dict={0:"玻璃种",1:"高冰种",2:"冰种",7:"冰糯种",3:"细糯种",4:"糯种",5:"细豆种",6:"豆种"}
                y_pre.append(trans_dict[pre_value])
                # train_label_dict={6:0,7:1,8:2,51:3,9:4,10:5,11:6,12:7}
                train_label_dict={6:"玻璃种",7:"高冰种",8:"冰种",51:"冰糯种",9:"细糯种",10:"糯种",11:"细豆种",12:"豆种"}
                y_label.append(train_label_dict[int(j)])

                # if j==str(7):
                #     source_image_path="{0}/{1}".format(check_image,"7")
                #     save_7_path="/home/ubuntu/Image/gqf_DCL-master/dataset/fujian_danmian_zhong_new/7"
                #     if not os.path.exists(save_7_path):
                #         os.makedirs(save_7_path)
                #     if i in os.listdir(source_image_path):
                #         shutil.copy(os.path.join(source_image_path,i),save_7_path)
                # elif j==str(51):
                #     source_image_path="{0}/{1}".format(check_image,"51")
                #     save_51_path="/home/ubuntu/Image/gqf_DCL-master/dataset/fujian_danmian_zhong_new/51"
                #     if not os.path.exists(save_51_path):
                #         os.makedirs(save_51_path)
                #     if i in os.listdir(source_image_path):
                #         shutil.copy(os.path.join(source_image_path,i),save_51_path)


                # image_source_path=os.path.join("/home/ubuntu/Image/gqf_DCL-master/dataset/fujian_danmian_zhong/source_image",j,i)
                # directin_path=os.path.join("/home/ubuntu/Image/gqf_DCL-master/dataset/fujian_danmian_zhong/save_danmian_zhong","tru_"+really_label_dict[int(j)],"pre_"+pre_label_dict[pre_value])
                # if not os.path.exists(directin_path):
                #     os.makedirs(directin_path)
                # shutil.copy(image_source_path,directin_path)



        # print(accuracy_score(y_label,y_pre,normalize=True))
        # print(recall_score(y_label,y_pre))
        # precision, recall, fscore, support = score(y_label, y_pre)
        #["玻璃种", "高冰种", "冰种","冰糯种", "细糯种","糯种","细豆种","豆种"]
        #["极透光","亚透光","半透光","微透光","不透光"]
        print(classification_report(y_label,y_pre,labels=["玻璃种", "高冰种", "冰种","冰糯种", "细糯种","糯种","细豆种","豆种"],digits=4))
        # print("precision：{}".format(precision))   target_names=["极透光", "亚透光", "半透光", "微透光"]
        # print("recall:{}".format(recall))
        # sns.set()
        # ax=plt.subplot()
        # C2 = confusion_matrix(y_label, y_pre, labels=["特浓","浓","适中","浅"])
        # C2 = confusion_matrix(y_label, y_pre, labels=["特浓", "浓", "适中", "浅"])
        # C2 = confusion_matrix(y_label, y_pre, labels=[0,1,2,3,4,5])
        C2=confusion_matrix(y_label, y_pre, labels=["玻璃种", "高冰种", "冰种","冰糯种", "细糯种","糯种","细豆种","豆种"])
        print(C2)

        # x, y = roc_draw(np.array(y_pre), np.array(y_label))
        # plt.plot(x,y)
        # plt.show()
        # plt.savefig("/home/ubuntu/Image/gqf_DCL-master/dataset/fujian_danmian_zhong/ROC/1.png")
        # dict_lei={}
        # blank_dict={}
        # head=["image_name","模型预测结果","磊哥打标结果"]
        # image_labels=os.listdir(check_image)
        # for image_label in image_labels:
        #     image_names=os.listdir(os.path.join(check_image,image_label))
        #     for image_name in image_names:
        #         dict_lei[image_name]=[image_label]
        # merge_dict=[dict_lei,dict_light]
        # print("light:",dict_light)
        # print("lei:",dict_lei)
        # for key in  dict_lei.keys():
        #
        #     blank_dict[key]=list(np.concatenate(list(d[key] for d in merge_dict)))
        # df = pd.DataFrame(blank_dict).T
        # print(df)
        # df.to_csv("/home/ubuntu/Image/gqf_DCL-master/dataset/fujian_shuozhuo_shui/打标统计结果.csv",encoding="utf-8")






