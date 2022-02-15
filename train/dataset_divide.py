import os
import cv2


# 训练集的保留比例
CAT_CLASS_RATIO = 1
DOG_CLASS_RATIO = 1


CAT_DATASET_NUM = 12500
DOG_DATASET_NUM = 12500


class DataSet_Deal:
    def __init__(self):
        pass

    def cat_dataset_divide(ratio):
        # 训练集划分
        for image_id in range(0, int(ratio * CAT_CLASS_RATIO)):
            image = cv2.imread(os.path.join("/home/wh/vscode/dog_vs_vat/train/cat.%s.jpg" % (image_id)))
            if not (os.path.exists("/home/wh/vscode/dog_vs_vat/data/cat/cat.%s.jpg" % image_id)):
                cv2.imwrite(os.path.join("/home/wh/vscode/dog_vs_vat/data/cat", "cat.%s.jpg" % (image_id)), image)

        print("--------训练集划分完成----------")

        # 测试集的划分
        for image_id in range(int(ratio * CAT_CLASS_RATIO), ratio):
            image = cv2.imread(os.path.join("/home/wh/vscode/dog_vs_vat/train/cat.%s.jpg" % (image_id)))
            if not (os.path.exists("/home/wh/vscode/dog_vs_vat/data/cat/cat.%s.jpg" % image_id)):
                cv2.imwrite(os.path.join("/home/wh/vscode/dog_vs_vat/data/cat", "cat.%s.jpg" % (image_id)), image)

        print("--------测试集划分完成----------")
        return

    def dog_dataset_divide(ratio):
        # 训练集划分
        for image_id in range(0, int(ratio * DOG_CLASS_RATIO)):
            image = cv2.imread(os.path.join("/home/wh/vscode/dog_vs_vat/train/cat.%s.jpg" % (image_id)))
            if not (os.path.exists("/home/wh/vscode/dog_vs_vat/data/dog/dog.%s.jpg" % image_id)):
                cv2.imwrite(os.path.join("/home/wh/vscode/dog_vs_vat/data/dog", "dog.%s.jpg" % (image_id)), image)

        print("--------训练集划分完成----------")

        # 测试集的划分
        for image_id in range(int(ratio * DOG_CLASS_RATIO), ratio):
            image = cv2.imread(os.path.join("/home/wh/vscode/dog_vs_vat/train/dog.%s.jpg" % (image_id)))
            if not (os.path.exists("/home/wh/vscode/dog_vs_vat/data/dog/dog.%s.jpg" % image_id)):
                cv2.imwrite(os.path.join("/home/wh/vscode/dog_vs_vat/data/dog", "dog.%s.jpg" % (image_id)), image)

        print("--------测试集划分完成----------")
        return

    # 创建文件夹函数
    def mkdir_folder(path):
        """
           param:
                path: 传入要创建文件夹的位置
        """
        folder = os.path.exists(path)

        if not folder:
            os.makedirs(path)
            print("... OK! ... Folder creation completed !!! ......")
        else:
            print("..... The folder to be created already exists! .....")


path_1 = "/home/wh/vscode/dog_vs_vat/data/cat"
path_2 = "/home/wh/vscode/dog_vs_vat/data/dog"
DataSet_Deal.mkdir_folder(path_1)
DataSet_Deal.mkdir_folder(path_2)


if __name__ == "__main__":
    DataSet_Deal.cat_dataset_divide(CAT_DATASET_NUM)
    DataSet_Deal.dog_dataset_divide(DOG_DATASET_NUM)
