# 本文件中主要定义了所有的通用文件操作方法
# 主要包括了对数据集的读取等
# 需要注意，数据集文件使用的是IDX文件格式，详细的解释可以参考数据集的网站或是https://blog.csdn.net/jiede1/article/details/77099326

import numpy as np
import struct


class OpData:
    @classmethod
    def read_idx1_file(cls, filepath:str):
        """
        读取数据集中IDX1格式的文件。
        :param filepath: 文件路径
        :return: ndarray, shape=(N,)
        """

        # 读取二进制的数据
        binary_data = open(filepath, "rb").read()

        # 解析文件头，不要纳入最终的数据中
        offset = 0
        fmt_header = ">ii"
        magic_number, num_images = struct.unpack_from(fmt_header, binary_data, offset)

        # Test
        # print("Magic Num=%d, Images Num=%d" % (magic_number, num_images))

        # 解析数据集
        offset += struct.calcsize(fmt_header)
        fmt_image = ">B"
        labels = np.empty(num_images)
        for i in range(num_images):
            labels[i] = struct.unpack_from(fmt_image, binary_data, offset)[0]
            offset += struct.calcsize(fmt_image)

        # 返回
        return labels


    @classmethod
    def read_idx3_file(cls, filepath: str):
        """
        读取数据集中的IDX3格式数据（灰度图像）
        :param filepath: 文件路径
        :return: ndarray, shape=(N, h*w)
        """

        binary_data = open(filepath, "rb").read()

        offset = 0
        fmt_header = ">iiii"
        magic_num, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, binary_data, offset)

        # print("Magic Num=%d, Images Num=%d, Rows=%d, Cols=%d" % (magic_num, num_images, num_rows, num_cols))

        image_size = num_rows * num_cols
        offset += struct.calcsize(fmt_header)
        fmt_image = ">" + str(image_size) + "B"
        images = np.empty((num_images, num_rows*num_cols))
        for i in range(num_images):
            images[i] = np.array(struct.unpack_from(fmt_image, binary_data, offset)).reshape((num_rows*num_cols))
            offset += struct.calcsize(fmt_image)

        return images


if __name__ == '__main__':
    import config
    CONFIG = config.Config()
    op_data = OpData()
    data = OpData.read_idx1_file(CONFIG.train_labels_path)
    # print(type(data))
    data3 = OpData.read_idx3_file(CONFIG.train_images_path)
    print(type(data3), data3.shape)
