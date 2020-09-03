import pandas as pd
import os
import shutil
import json
import numpy as np

def adjust_image(image_path,origin_data_file,model="train"):
    '''
    划分训练集，验证集，和测试集
    :param image_path:待划分图片保存位置
    :param origin_data_file:原始数据表格保存位置
    :param model:模式
    :return:
    '''
    image_name = os.listdir(image_path)
    nums = len(image_name)
    # print(":",nums)
    train_list = image_name[:int(nums * 0.8)]
    val_list = image_name[int(nums * 0.8):int(nums * 0.9)]
    test_list = image_name[int(nums * 0.9):int(nums * 1.0)]

    if model=="train":
        wp=open(os.path.join(os.path.dirname(image_path),str("anno"),"{}.txt".format(model)),"a")
        move_image(image_path,train_list,origin_data_file,wp,model)
        wp.close()
    if model=="test":
        wp = open(os.path.join(os.path.dirname(image_path), str("anno"), "{}.txt".format(model)),"a")
        move_image(image_path,test_list,origin_data_file,wp,model)
        wp.close()
    if model=="val":
        wp = open(os.path.join(os.path.dirname(image_path), str("anno"), "{}.txt".format(model)),"a")
        move_image(image_path,val_list,origin_data_file,wp,model)
        wp.close()


def move_image(image_path,data_list,origin_data_file,wp,model):
    '''
    根据图片列表与每个图片的标签和保存路径，依次保存到相应路径并写入标签
    :param image_path:图片路径，方便保存位置及标签判断函数调用
    :param data_list:相应模式（训练，验证，测试）下的数据列表
    :param origin_data_file:原始数据表格，方便保存位置及标签判断函数调用
    :param wp:文档指针
    :param model:
    :return:
    '''
    num=0
    count=0
    for image_name in data_list:
        direction_path,adjust_label=location_judge(image_path,image_name,origin_data_file,model)
        if isinstance(direction_path,str):
            if not os.path.exists(direction_path):
                os.makedirs(direction_path)
            shutil.copy(os.path.join(image_path,image_name),direction_path)
            wp.write(str(os.path.join(direction_path,image_name))+" "+str(adjust_label)+"\n")
            num+=1
            print("拷贝成功",model,num)
            # print(os.path.join(image_path,image_name))
            # print(os.path.join(direction_path,image_name))
            # print("保存路径：",direction_path,"图片标签：",adjust_label)
        else:
            count+=1
            print("找不到该图片",count)
    print("num:",num,"count:",count)


def location_judge(image_path,image_name,origin_data_file,model):
    '''
    调用图片标签字典，并将标签列表转为0，1，2..，方便训练，根据图片名在字典中对应的键确定最终图片保存位置及标签
    :param image_path: 原图片的存放位置
    :param image_name: 每一个图片名
    :param origin_data_file: 图片的原始数据，包含图片的名字和标签等信息
    :param model: 确定最终保存在训练，测试，还是验证集中
    :return: 返回最终的标签和保存地址
    注意：图片列表键所对应的值和保存位置及返回的标签要严格对应
    '''
    image_label_dict = get_label_list_dict(origin_data_file)
    # print(image_label_dict[6])
    "种标签和保存位置"
    # label_list = list([6, 7, 8, 9, 10, 11, 12, 51])
    # label2index = dict(zip(sorted(set(label_list)), list(range(len(label_list)))))
    # if image_name in image_label_dict[6]:
    #     direction_path=os.path.join(os.path.dirname(image_path),str("data"),str(model),str("6"))
    #     adjust_label= label2index[6]
    # elif image_name in image_label_dict[7]:
    #     direction_path = os.path.join(os.path.dirname(image_path), str("data"), str(model), str("7"))
    #     adjust_label = label2index[7]
    # elif image_name in image_label_dict[8]:
    #     direction_path = os.path.join(os.path.dirname(image_path), str("data"), str(model), str("8"))
    #     adjust_label = label2index[8]
    # elif image_name in image_label_dict[9]:
    #     direction_path = os.path.join(os.path.dirname(image_path), str("data"), str(model), str("9"))
    #     adjust_label = label2index[9]
    # elif image_name in image_label_dict[10]:
    #     direction_path = os.path.join(os.path.dirname(image_path), str("data"), str(model), str("10"))
    #     adjust_label = label2index[10]
    # elif image_name in image_label_dict[11]:
    #     direction_path = os.path.join(os.path.dirname(image_path), str("data"), str(model), str("11"))
    #     adjust_label = label2index[11]
    # elif image_name in image_label_dict[12]:
    #     direction_path = os.path.join(os.path.dirname(image_path), str("data"), str(model), str("12"))
    #     adjust_label = label2index[12]
    # elif image_name in image_label_dict[51]:
    #     direction_path = os.path.join(os.path.dirname(image_path), str("data"), str(model), str("51"))
    #     adjust_label = label2index[51]
    "谁标签和保存位置"
    label_list = list([13,14,15,16,17])
    label2index = dict(zip(sorted(set(label_list)), list(range(len(label_list)))))
    if image_name in image_label_dict[13]:
        direction_path=os.path.join(os.path.dirname(image_path),str("data"),str(model),str("13"))
        adjust_label= label2index[13]
    elif image_name in image_label_dict[14]:
        direction_path = os.path.join(os.path.dirname(image_path), str("data"), str(model), str("14"))
        adjust_label = label2index[14]
    elif image_name in image_label_dict[15]:
        direction_path = os.path.join(os.path.dirname(image_path), str("data"), str(model), str("15"))
        adjust_label = label2index[15]
    elif image_name in image_label_dict[16]:
        direction_path = os.path.join(os.path.dirname(image_path), str("data"), str(model), str("16"))
        adjust_label = label2index[16]
    elif image_name in image_label_dict[17]:
        direction_path = os.path.join(os.path.dirname(image_path), str("data"), str(model), str("17"))
        adjust_label = label2index[17]

    else:
        direction_path=np.nan
        adjust_label=np.nan

    return direction_path,adjust_label


def get_label_list_dict(origin_data_file):
    """
    与下面函数get_label_dict类似，都是由原始文件得到键为标签，值为标签所对应的图片名字列表，但不同的是该函数
    原始文件下的种标签不是字典格式，而是单一的数字,和下面的get_label_dict只用其一。
    :param orgin_data_file: 图片的原始数据，包含图片的名字和标签等信息
    :return: 返回字典，以便调用
    """
    df = pd.read_csv(open(origin_data_file))
    names = df["local_image_url"]
    # zhong_label = df["zhong"]

    shui_label=df["grade_property_json"]

    # shui_label=df["shui"]
    name_list=[]
    anno_dict={}
    # zhong_label_list=[]
    shui_label_list=[]
    # for names,zhong_label in list(zip(names,zhong_label)):
    for names, shui_label in list(zip(names, shui_label)):
        if "shui" in str(shui_label):
            shui_label=json.loads(shui_label)["base_data"]
            shui_label=shui_label["shui"]
            # print(type(shui_label))
            # print(type(names))
            # if (isinstance(zhong_label,float)) & (isinstance(names,str)) & ("gemImg" in names):
            if (isinstance(shui_label, int)) & (isinstance(names, str)) & ("gemImg" in names):
                name = json.loads(names)
                img_name = name[0]["url"].split("/")[-1]
                name_list.append(img_name)
                # zhong_label_list.append(zhong_label)
                shui_label_list.append(shui_label)

    # for img, anno in zip(name_list, zhong_label_list):
    for img, anno in zip(name_list, shui_label_list):
        if not anno in anno_dict:
            anno_dict[anno] = [img]
        else:
            anno_dict[anno].append(img)
    # print(anno_dict)
    # print(anno_dict.keys())
    return anno_dict



def get_label_dict(origin_data_file):
    '''
    生成一个字典，图片标签为键，同一个标签下的图片名组成一个列表作为值，有几个标签就对应着几个键值对
    :param origin_data_file: 图片的原始数据，包含图片的名字和标签等信息
    :return: 返回字典，以便调用
    '''
    df = pd.read_csv(open(origin_data_file))
    names = df["local_image_url"]
    zhong_label = df["grade_property_json"]
    name_list=[]
    anno_dict={}
    zhong_label_list=[]
    for names, zhong_label in list(zip(names, zhong_label)):
        if isinstance(zhong_label, str) & isinstance(names, str):
            if (str("zhong") in zhong_label) & ("gemImg" in names):
                name = json.loads(names)
                zhong_label=json.loads(zhong_label)

                #如果每个id下只取一个名字，则不需要下面的循环
                for image_names in name:
                    img_name=image_names["url"]
                    img_label=zhong_label["base_data"]["zhong"]
                    img_name=img_name.split("/")[-1]
                    name_list.append(img_name)
                    zhong_label_list.append(img_label)
    # print(name_list)
    for img, anno in zip(name_list, zhong_label_list):
        if not anno in anno_dict:
            # print(img)
            anno_dict[anno] = [img]
        else:
            anno_dict[anno].append(img)
    return anno_dict


if __name__ == '__main__':
    wushipai_source_path=r"/home/ubuntu/Image/gqf_DCL-master/dataset/wushipai_class_zhong/无事牌种统一.csv"
    wushipai_image_path=r"/home/ubuntu/Image/gqf_DCL-master/dataset/wushipai_class_zhong/no_higlt_orisize_image"
    class_zhong_source_path=r"/home/ubuntu/Image/gqf_DCL-master/dataset/class_zhong/种意见统一.csv"
    class_zhong_image_path=r"/home/ubuntu/Image/gqf_DCL-master/dataset/class_zhong/source_image"

    ruyi_image_path=r"/home/ubuntu/Image/gqf_DCL-master/dataset/ruyi_zhong/no_highlt_image"
    ruyi_source_file_path=r"dataset/ruyi_zhong/如意种一致.csv"

    pingankou_zhong_image_path=r"/home/ubuntu/Image/gqf_DCL-master/dataset/pingankou_zhong/no_highlt_image"
    pingankou_zhong_source_file_path=r"/home/ubuntu/Image/gqf_DCL-master/dataset/pingankou_zhong/平安扣种一致.csv"

    pingankou_shui_image_path=r"/home/ubuntu/Image/gqf_DCL-master/dataset/pingankou_shui/no_highlt_image"
    pingankou_shui_source_file=r"/home/ubuntu/Image/gqf_DCL-master/dataset/pingankou_shui/平安扣水一致.csv"

    # class_zhong_source_image=r"/home/ubuntu/Image/gqf_DCL-master/dataset/class_zhong/source_image"


    dan_mian_shui_file=r"/home/ubuntu/Image/gqf_DCL-master/dataset/dan_mian_shui/蛋面水意见统一.csv"
    dan_mian_shui_image=r"/home/ubuntu/Image/gqf_DCL-master/dataset/dan_mian_shui/no_highlt_image"
    adjust_image(dan_mian_shui_image, dan_mian_shui_file, model="test")
    # get_label_list_dict(dan_mian_shui_file)
    # get_label_list_dict(ruyi_source_file_path)
    # get_label_dict(class_zhong_source_path)
    # get_label_list_dict(pingankou_zhong_source_file_path)

