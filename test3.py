import os
import string
import cv2
source_image = "/home/ubuntu/Image/gqf_DCL-master/dataset/fujian_guanyin_shui/source_image" #这里改成自己的图片所在路径

labels=os.listdir(source_image)
count=0
for label in labels:
    image_paths=os.path.join(source_image,label)
    # print(image_paths)
    image_names=os.listdir(image_paths)
    for image_name in image_names:

        source_image_path=os.path.join(source_image,label,image_name)


        image_name=image_name.replace("jpg",".jpg")
        image_name=image_name.replace("..jpg",".jpg")

        image_name=image_name.replace("JPG",".JPG")
        image_name=image_name.replace("..JPG",".JPG")

        image_name=image_name.replace("png",".png")
        image_name=image_name.replace("..png",".png")

        image_name=image_name.replace("PNG",".PNG")
        image_name=image_name.replace("..PNG",".PNG")

        image_name=image_name.replace("jpeg",".jpeg")
        image_name=image_name.replace("..jpeg",".jpeg")
        new_image_path=os.path.join(source_image,label,image_name)

        os.rename( source_image_path,new_image_path)
        count+=1
        print("更新成功：",count)

