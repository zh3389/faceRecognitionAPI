# 识别并在面上绘制方框
import face_recognition
from PIL import Image, ImageDraw
import numpy as np

# 这是在单个图像上运行人脸识别的示例并在每个被识别的人周围画一个方框。

# 加载一张示例图片并学习如何识别它。
obama_image = face_recognition.load_image_file("./images/obama.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# 加载第二张示例图片并学习如何识别它。
biden_image = face_recognition.load_image_file("./images/biden.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

# 创建已知人脸编码及其名称的数组。
known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding
]
known_face_names = [
    "Barack Obama",
    "Joe Biden"
]

# 加载具有未知人脸的图像
unknown_image = face_recognition.load_image_file("./images/two_people.jpg")

# 查找未知图像中的所有人脸和人脸编码
face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

# 将图像转换为PIL格式的图像，以便我们可以使用Pillow库在其顶部绘制
# See http://pillow.readthedocs.io/ for more about PIL/Pillow
pil_image = Image.fromarray(unknown_image)
# 创建要绘制的Pillow ImageDraw Draw实例
draw = ImageDraw.Draw(pil_image)

# 循环浏览未知图像中发现的每个人脸
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    # 查看该面是否与已知面匹配
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

    name = "Unknown"

    # 如果在known_face_encodings中找到匹配项，请使用第一个。
    # if True in matches:
    #     first_match_index = matches.index(True)
    #     name = known_face_names[first_match_index]

    # 或者，使用与新面的距离最小的已知面
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = known_face_names[best_match_index]

    # 使用枕头模块在面部周围画一个方框
    draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

    # 在面下方绘制一个带有名称的标签
    text_width, text_height = draw.textsize(name)
    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
    draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))


# R根据Pillow文档从内存中删除绘图库
del draw

# 显示生成的图像
pil_image.show()

# 如果需要，还可以通过取消注释此行将新映像的副本保存到磁盘
# pil_image.save("image_with_boxes.jpg")
exit()
