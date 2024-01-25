import face_recognition


class Recognition:
    def __init__(self):
        pass

    def recognizeEncoding(self, image):
        """
        人脸识别 返回第一张人脸的 128 维编码
        :param image: 输入一张RGB的 numpy.ndarray 图片
        :return: 其中一张人脸的 128 维 encoding [ 0.00212635  0.18151696 ...  0.08942952 -0.02890663]
        """
        # image = face_recognition.load_image_file("./images/biden.jpg")  # = PIL.Image.open
        face_encoding = face_recognition.face_encodings(image, model="cnn")[0]
        return face_encoding

    def recognizeCompareImage(self, image1, image2):
        """
        人脸比较 返回比较结果 True False
        :param image1: 输入一张RGB的 numpy.ndarray 图片
        :param image2: 输入一张RGB的 numpy.ndarray 图片
        :return: 返回比较结果 [True] or [False]
        """
        # image1 = face_recognition.load_image_file("./images/biden.jpg")  # = np.array(PIL.Image.open)
        # image2 = face_recognition.load_image_file("./images/obama.jpg")  # = np.array(PIL.Image.open)
        encoding1 = face_recognition.face_encodings(image1, model="cnn")[0]
        encoding2 = face_recognition.face_encodings(image2, model="cnn")[0]
        results = face_recognition.compare_faces([encoding1], encoding2)  # True False
        return results

    def recognizeCompareEncoding(self, encoding_list, encoding, tolerance=0.5, show=False):
        """
        人脸比较 返回比较结果 True False
        :param encoding_list: [encoding1, encoding2, encoding3, ...]
        :param encoding: [ 0.00212635  0.18151696 ...  0.08942952 -0.02890663]
        :param show: 是否打印比较结果
        :return: 返回比较结果
        """
        face_distances = face_recognition.face_distance(encoding_list, encoding)
        if show:
            for i, face_distance in enumerate(face_distances):
                print("The test image has a distance of {:.2} from known image #{}".format(face_distance, i))
                print("- With a normal cutoff of 0.6, would the test image match the known image? {}".format(
                    face_distance < 0.6))
                print("- With a very strict cutoff of 0.5, would the test image match the known image? {}".format(
                    face_distance < 0.5))
                print()
        return face_distances


if __name__ == '__main__':
    recognition = Recognition()
    # image = face_recognition.load_image_file("./images/biden.jpg")
    image = face_recognition.load_image_file("/app/utils/images/biden.jpg")
    face_encoding = recognition.recognizeEncoding(image)
    print(face_encoding)
    res = recognition.recognizeCompareImage(image, image)
    print(res)
    res = recognition.recognizeCompareEncoding([face_encoding], face_encoding, show=True)
    print(res)
