import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from function.utils import *

#### 调整image&&xml大小到目标大小

# 输入图像文件夹和输出图像文件夹 imgnet_2
input_folder = '/data/dataset/imagenet_2/tiny_train/n02087046'
output_folder = "/data/dataset/imagenet_2/test1/n02087046"

input_xml = '/data/dataset/imagenet_2/bbox/n02087046'
output_xml = "/data/dataset/imagenet_2/testbox/n02087046"

target_size = (224, 224)





def resavexml():
    if not os.path.exists(output_xml):
        os.makedirs(output_xml)

    x = input_xml
    xml_files = os.listdir(x)
    for xml_file in xml_files:
        xml_path = os.path.join(x, xml_file)
        bounding_boxes = load_parse_xml(xml_path)
        if len(bounding_boxes) > 1:  #single target only
            continue
        orgname = xml_file.split('.')[-2]
        orgfile = os.path.join(input_folder, orgname+'.JPEG')
        if not os.path.exists(orgfile):
            continue
        orgimg = Image.open(orgfile)
        org_size = orgimg.size
        # 裁剪偏移量
        crop_offset = ((org_size[0] - target_size[0]) // 2, (org_size[1] - target_size[1]) // 2)
        # 调整bounding boxes
        adjusted_bounding_boxes = [adjust_bounding_box(bbox, crop_offset, target_size, org_size) for bbox in bounding_boxes]
        if adjusted_bounding_boxes[0][2]-adjusted_bounding_boxes[0][0] >200 and adjusted_bounding_boxes[0][3]-adjusted_bounding_boxes[0][1]>200:
            continue
        xml_newpath = os.path.join(output_xml, xml_file)
        save_parse_xml(adjusted_bounding_boxes[0],xml_newpath)

    print("xml处理完成并保存到", output_xml)


def resizePic():
    # 定义图像转换
    transform = transforms.Compose([
        transforms.CenterCrop(target_size),  # 调整大小到目标大小
    ])

    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取输入文件夹中的所有图像文件
    image_files = os.listdir(input_folder)

    # 对每张图像进行处理并保存
    for image_file in image_files:
        name = image_file.split('.')[-2]
        xml_path = os.path.join(output_xml, name+'.xml')
        if not os.path.exists(xml_path):
            continue
        # 打开图像
        image_path = os.path.join(input_folder, image_file)
        image = Image.open(image_path)
        # 应用图像转换
        transformed_image = transform(image)
        # 构造输出图像路径
        output_path = os.path.join(output_folder, image_file)
        # 保存图像
        transformed_image.save(output_path)
    print("图像处理完成并保存到", output_folder)


if __name__ == '__main__':
    # resavexml()
    resizePic()
    
