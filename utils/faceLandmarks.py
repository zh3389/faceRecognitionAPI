import numpy as np
import face_recognition
from PIL import Image, ImageDraw


class Landmarks:
    def __init__(self):
        pass

    def landmarksPoint(self, image):
        """
        人脸地标检测 获取图像中的人脸地标
        :param image: 输入一张RGB的 numpy.ndarray 图片
        :return: 所有人脸地标 [{'chin': [(429, 328), ..., (707, 382)],
                              'left_eyebrow': [(488, 294), (509, 279), (535, 278), (561, 283), (584, 296)],
                              'right_eyebrow': [(622, 307), (646, 305), (670, 309), (691, 321), (698, 344)],
                              'nose_bridge': [(601, 328), (599, 352), (598, 375), (596, 400)],
                              'nose_tip': [(555, 414), (570, 421), (586, 428), (601, 428), (614, 426)],
                              'left_eye': [(512, 320), (528, 316), (544, 319), (557, 331), (541, 330), (525, 327)],
                              'right_eye': [(629, 348), (647, 342), (661, 346), (672, 357), (659, 358), (644, 354)],
                              'top_lip': [(519, 459), ..., (527, 459)],
                              'bottom_lip': [(627, 480), ..., (620, 477)]}]
        """
        # image = face_recognition.load_image_file("./images/biden.jpg")  # = np.array(PIL.Image.open)
        face_landmarks_list = face_recognition.face_landmarks(image)
        return face_landmarks_list

    def landmarksDraw(self, image):
        """
        人脸地标检测 在源图像中绘制人脸地标 以数组返回
        :param image: 输入一张RGB的 numpy.ndarray 图片
        :return: 在原图上绘制检测到的人脸地标，以 numpy.ndarray 输出
        """
        # image = face_recognition.load_image_file("./images/two_people.jpg")  # = np.array(PIL.Image.open)
        drawImage = Image.fromarray(image)
        face_landmarks_list = face_recognition.face_landmarks(image)
        draw = ImageDraw.Draw(drawImage)
        for face_landmarks in face_landmarks_list:
            for facial_feature in face_landmarks.keys():
                draw.line(face_landmarks[facial_feature], width=5)
        del draw
        # drawImage.show()
        return np.array(drawImage)

    def landmarksMakeup(self, image):
        """
        人脸地标检测 在源图像中绘制妆容 以数组返回
        :param image: 输入一张RGB的 numpy.ndarray 图片
        :return: 在原图上通过人脸地标绘制妆容，以 numpy.ndarray 输出
        """
        # image = face_recognition.load_image_file("./images/biden.jpg")  # = np.array(PIL.Image.open)
        face_landmarks_list = face_recognition.face_landmarks(image)
        drawImage = Image.fromarray(image)
        for face_landmarks in face_landmarks_list:
            d = ImageDraw.Draw(drawImage, 'RGBA')

            # Make the eyebrows into a nightmare
            d.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 128))
            d.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 128))
            d.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 150), width=5)
            d.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 150), width=5)

            # Gloss the lips
            d.polygon(face_landmarks['top_lip'], fill=(150, 0, 0, 128))
            d.polygon(face_landmarks['bottom_lip'], fill=(150, 0, 0, 128))
            d.line(face_landmarks['top_lip'], fill=(150, 0, 0, 64), width=8)
            d.line(face_landmarks['bottom_lip'], fill=(150, 0, 0, 64), width=8)

            # Sparkle the eyes
            d.polygon(face_landmarks['left_eye'], fill=(255, 255, 255, 30))
            d.polygon(face_landmarks['right_eye'], fill=(255, 255, 255, 30))

            # Apply some eyeliner
            d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(0, 0, 0, 110), width=6)
            d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(0, 0, 0, 110), width=6)

        del d
        # drawImage.show()
        return np.array(drawImage)


if __name__ == '__main__':
    landmark = Landmarks()
    image = Image.open("./images/biden.jpg")
    image = np.array(image)
    res = landmark.landmarksPoint(image)
    print(res)
    res = landmark.landmarksDraw(image)
    print(res)
    res = landmark.landmarksMakeup(image)
    print(res)
