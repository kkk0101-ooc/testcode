import xml.etree.ElementTree as ET

def load_parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    bounding_boxes = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        bounding_boxes.append((xmin, ymin, xmax, ymax))

    return bounding_boxes


def save_parse_xml(bounding_boxes,dir):

    # 创建根元素
    root = ET.Element("annotation")
    subroot = ET.SubElement(root, "object")
    ssubroot = ET.SubElement(subroot, "bndbox")
    nameroot = ET.SubElement(subroot, "name")
    nameroot.text = dir.split('/')[-1].split('.')[-2]

    # 创建子元素
    item1 = ET.SubElement(ssubroot, "xmin")
    item1.text = str(bounding_boxes[0])

    item2 = ET.SubElement(ssubroot, "ymin")
    item2.text = str(bounding_boxes[1])

    item3 = ET.SubElement(ssubroot, "xmax")
    item3.text = str(bounding_boxes[2])

    item4 = ET.SubElement(ssubroot, "ymax")
    item4.text = str(bounding_boxes[3])

    # 将根元素转换为 ElementTree 对象
    tree = ET.ElementTree(root)

    # 将 ElementTree 对象写入 XML 文件
    tree.write(dir)




def adjust_bounding_box(bbox, crop_offset, crop_size, original_size):
    # bbox: 原始边界框坐标 (x_min, y_min, x_max, y_max)
    # crop_offset: 裁剪偏移量 (crop_offset_x, crop_offset_y)
    # crop_size: 裁剪尺寸 (crop_width, crop_height)
    # original_size: 原始图像尺寸 (original_width, original_height)
    
    # 计算裁剪后的边界框坐标
    x_min = max(0, bbox[0] - crop_offset[0])
    y_min = max(0, bbox[1] - crop_offset[1])
    x_max = min(crop_size[0], bbox[2] - crop_offset[0])
    y_max = min(crop_size[1], bbox[3] - crop_offset[1])

    return (x_min, y_min, x_max, y_max)
