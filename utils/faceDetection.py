import numpy as np
import face_recognition
from PIL import Image, ImageDraw


class Detection:
    def __init__(self):
        pass

    def detectBox(self, image):
        """
        获取人脸检测框 左上角坐标 和 右下角坐标 返回坐标列表
        :param image: 输入一张RGB的 numpy.ndarray 图片
        :return: 所有人脸坐标 [(241, 740, 562, 419), (Top, Left, Bottom, Right)]
        """
        # image = face_recognition.load_image_file("./images/biden.jpg")  # = np.array(PIL.Image.open)
        box_list = face_recognition.face_locations(image)
        return box_list

    def detectFace(self, image):
        """
        获取人脸检测数组，每个数组为一张人脸 返回数组列表
        :param image: 输入一张RGB的 numpy.ndarray 图片
        :return: 所有人脸 numpy.ndarray [array([[[133, 106,  79], [138, 108,  84], [139, 109,  85], ...,
                                               [ 51, 101, 150], [ 55, 109, 156], [ 48, 104, 151]]], dtype=uint8)]
        """
        # image = face_recognition.load_image_file("./images/biden.jpg")  # = np.array(PIL.Image.open)
        face_locations = face_recognition.face_locations(image)
        face_list = []
        for face_location in face_locations:
            top, right, bottom, left = face_location
            face_image = image[top:bottom, left:right]
            face_list.append(face_image)
        return face_list

    def detectDrawBox(self, image):
        """
        获取人脸检测并绘制在原图 返回原图数组
        :param image: 输入一张RGB的 numpy.ndarray 图片
        :return: 在原图上绘制检测到的人脸框，以 numpy.ndarray 输出
        """
        # image = face_recognition.load_image_file("./images/biden.jpg")  # = np.array(PIL.Image.open)
        drawImage = Image.fromarray(image)
        draw = ImageDraw.Draw(drawImage)
        face_locations = face_recognition.face_locations(image)
        for face_location in face_locations:
            name = "Face"
            top, right, bottom, left = face_location
            draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
            text_width, text_height = draw.textsize(name)
            draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
            draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))
        del draw
        drawImage.show()
        # drawImage.save("drawBoxSourceImg.jpg")
        return np.array(drawImage)


if __name__ == '__main__':
    detection = Detection()
    image = Image.open("./images/biden.jpg")
    image = np.array(image)
    res = detection.detectBox(image)
    print(res)
    res = detection.detectFace(image)
    print(res)
    res = detection.detectDrawBox(image)
    print(res)
