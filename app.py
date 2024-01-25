import json
import numpy as np
from io import BytesIO
from typing import List
import face_recognition
from PIL import Image, ImageDraw
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi import FastAPI, Request, File, UploadFile

app = FastAPI()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """捕获默认框架返回的异常信息格式，修改为规定格式"""
    return JSONResponse({"success": False,
                         "code": 400,
                         "msg": f"接口参数传递错误",
                         "data": exc.errors()}
                        )


@app.post("/faceapi/v1/detectBox")
async def upload_image(imgFile: UploadFile = File(...)):
    """
    获取人脸检测框 左上角坐标 和 右下角坐标 返回坐标列表
    :param image: 上传一张图片
    :return: 所有人脸坐标 [(241, 740, 562, 419), (Top, Left, Bottom, Right)]
    """
    try:
        image_array = face_recognition.load_image_file(BytesIO(await imgFile.read()))
        if image_array is None:  # 确认图片是否成功读取
            raise ValueError("图片读取失败！")
        # image_array = face_recognition.load_image_file("./images/biden.jpg")  # = np.array(PIL.Image.open)
        face_boxes = face_recognition.face_locations(image_array)
        return {"success": True,
                "code": 200,
                "msg": f"人脸检测结果已返回，格式为: [(123, 123, 456, 456), (Top, Left, Bottom, Right)]...",
                "data": {"boxes": face_boxes}
                }
    except Exception as e:
        return {"success": False,
                "code": 500,
                "msg": str(e),
                "data": None}


@app.post("/faceapi/v1/detectFace")
async def upload_image(imgFile: UploadFile = File(...)):
    """
    获取人脸检测数组，每个数组为一张人脸 返回数组列表
    :param image: 上传一张图片
    :return: 所有人脸 json.dumps(numpy.ndarray.tolist())
             [array([[[133, 106,  79], [138, 108,  84], [139, 109,  85], ...,
             [ 51, 101, 150], [ 55, 109, 156], [ 48, 104, 151]]], dtype=uint8)]
    """
    try:
        image_array = face_recognition.load_image_file(BytesIO(await imgFile.read()))
        if image_array is None:  # 确认图片是否成功读取
            raise ValueError("图片读取失败！")
        # image_array = face_recognition.load_image_file("./images/biden.jpg")  # = np.array(PIL.Image.open)
        face_locations = face_recognition.face_locations(image_array)
        face_list = []
        for face_location in face_locations:
            top, right, bottom, left = face_location
            face_image = image_array[top:bottom, left:right]
            face_list.append(json.dumps(face_image.tolist()))
        return {"success": True,
                "code": 200,
                "msg": f"已返回所有人脸",
                "data": {"face_list": face_list}
                }
    except Exception as e:
        return {"success": False,
                "code": 500,
                "msg": str(e),
                "data": None}


@app.post("/faceapi/v1/detectDrawBox")
async def upload_image(imgFile: UploadFile = File(...)):
    """
    获取人脸检测并绘制在原图 返回原图数组
    :param image: 上传一张图片
    :return: 在原图上绘制检测到的人脸框，以 json.dumps(numpy.ndarray.tolist()) 输出
    """
    try:
        image_array = face_recognition.load_image_file(BytesIO(await imgFile.read()))
        if image_array is None:  # 确认图片是否成功读取
            raise ValueError("图片读取失败！")
        # image_array = face_recognition.load_image_file("./images/biden.jpg")  # = np.array(PIL.Image.open)
        drawImage = Image.fromarray(image_array)
        draw = ImageDraw.Draw(drawImage)
        face_locations = face_recognition.face_locations(image_array)
        for face_location in face_locations:
            top, right, bottom, left = face_location
            draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
            text_width, text_height = draw.textsize("Face")
            draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
            draw.text((left + 6, bottom - text_height - 5), "Face", fill=(255, 255, 255, 255))
        del draw
        # drawImage.show()
        # drawImage.save("drawBoxSourceImg.jpg")
        drawBoxFaceArray = json.dumps(np.array(drawImage).tolist())
        return {"success": True,
                "code": 200,
                "msg": f"在原图上绘制检测到的人脸框，以 numpy.ndarray 输出",
                "data": {"draw_image": drawBoxFaceArray}
                }
    except Exception as e:
        return {"success": False,
                "code": 500,
                "msg": str(e),
                "data": None}


@app.post("/faceapi/v1/landmarksPoint")
async def upload_image(imgFile: UploadFile = File(...)):
    """
    人脸地标检测 获取图像中的人脸地标
    :param image: 上传一张图片
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
    try:
        image_array = face_recognition.load_image_file(BytesIO(await imgFile.read()))
        if image_array is None:  # 确认图片是否成功读取
            raise ValueError("图片读取失败！")
        # image_array = face_recognition.load_image_file("./images/biden.jpg")  # = np.array(PIL.Image.open)
        face_landmarks_list = face_recognition.face_landmarks(image_array)
        return {"success": True,
                "code": 200,
                "msg": f"""已返回所有人脸地标""",
                "data": {"face_landmarks_list": face_landmarks_list}
                }
    except Exception as e:
        return {"success": False,
                "code": 500,
                "msg": str(e),
                "data": None}


@app.post("/faceapi/v1/landmarksDraw")
async def upload_image(imgFile: UploadFile = File(...)):
    """
    人脸地标检测 在源图像中绘制人脸地标 以数组返回
    :param image: 上传一张图片
    :return: 在原图上绘制检测到的人脸地标，以 json.dumps(numpy.ndarray.tolist()) 输出
    """
    try:
        image_array = face_recognition.load_image_file(BytesIO(await imgFile.read()))
        if image_array is None:  # 确认图片是否成功读取
            raise ValueError("图片读取失败！")
        # image_array = face_recognition.load_image_file("./images/two_people.jpg")  # = np.array(PIL.Image.open)
        drawImage = Image.fromarray(image_array)
        face_landmarks_list = face_recognition.face_landmarks(image_array)
        draw = ImageDraw.Draw(drawImage)
        for face_landmarks in face_landmarks_list:
            for facial_feature in face_landmarks.keys():
                draw.line(face_landmarks[facial_feature], width=5)
        del draw
        # drawImage.show()
        drawLandmarksFaceArray = json.dumps(np.array(drawImage).tolist())
        return {"success": True,
                "code": 200,
                "msg": f"已返回在原图上绘制检测到的人脸地标",
                "data": {"draw_image": drawLandmarksFaceArray}
                }
    except Exception as e:
        return {"success": False,
                "code": 500,
                "msg": str(e),
                "data": None}


@app.post("/faceapi/v1/landmarksMakeup")
async def upload_image(imgFile: UploadFile = File(...)):
    """
    人脸地标检测 在源图像中绘制妆容 以数组返回
    :param image: 上传一张图片
    :return: 在原图上通过人脸地标绘制妆容，以 json.dumps(numpy.ndarray.tolist()) 输出
    """
    try:
        image_array = face_recognition.load_image_file(BytesIO(await imgFile.read()))
        if image_array is None:  # 确认图片是否成功读取
            raise ValueError("图片读取失败！")
        # image_array = face_recognition.load_image_file("./images/biden.jpg")  # = np.array(PIL.Image.open)
        face_landmarks_list = face_recognition.face_landmarks(image_array)
        drawImage = Image.fromarray(image_array)
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
        drawMakeupFaceArray = json.dumps(np.array(drawImage).tolist())
        return {"success": True,
                "code": 200,
                "msg": f"已返回在原图上通过人脸地标绘制妆容",
                "data": {"draw_image": drawMakeupFaceArray}
                }
    except Exception as e:
        return {"success": False,
                "code": 500,
                "msg": str(e),
                "data": None}


@app.post("/faceapi/v1/recognizeEncoding")
async def upload_image(imgFile: UploadFile = File(...)):
    """
    人脸识别 返回第一张人脸的 128 维编码
    :param image: 上传一张图片
    :return: 其中一张人脸的 128 维 encoding [ 0.00212635  0.18151696 ...  0.08942952 -0.02890663]
    """
    try:
        image_array = face_recognition.load_image_file(BytesIO(await imgFile.read()))
        if image_array is None:  # 确认图片是否成功读取
            raise ValueError("图片读取失败！")
        # image_array = face_recognition.load_image_file("./images/biden.jpg")  # = PIL.Image.open
        face_encoding = face_recognition.face_encodings(image_array, model="cnn")[0]
        face_encoding = np.array(face_encoding).tolist()
        return {"success": True,
                "code": 200,
                "msg": f"其中一张人脸的 128 维 encoding [ 0.00212635  0.18151696 ...  0.08942952 -0.02890663]",
                "data": {"encoding": face_encoding}
                }
    except Exception as e:
        return {"success": False,
                "code": 500,
                "msg": str(e),
                "data": None}


@app.post("/faceapi/v1/recognizeCompareImage")
async def upload_image(imgFile1: UploadFile = File(...), imgFile2: UploadFile = File(...)):
    """
    人脸比较 返回比较结果 True False
    :param image1: 上传一张图片A
    :param image2: 上传一张图片B
    :return: 返回比较结果 [True] or [False]
    """
    try:
        image_array1 = face_recognition.load_image_file(BytesIO(await imgFile1.read()))
        if image_array1 is None:  # 确认图片是否成功读取
            raise ValueError("图片读取失败！")
        image_array2 = face_recognition.load_image_file(BytesIO(await imgFile2.read()))
        if image_array2 is None:  # 确认图片是否成功读取
            raise ValueError("图片读取失败！")
        # image_array1 = face_recognition.load_image_file("./images/biden.jpg")  # = np.array(PIL.Image.open)
        # image_array2 = face_recognition.load_image_file("./images/obama.jpg")  # = np.array(PIL.Image.open)
        encoding1 = face_recognition.face_encodings(image_array1, model="cnn")[0]
        encoding2 = face_recognition.face_encodings(image_array2, model="cnn")[0]
        results = face_recognition.compare_faces([encoding1], encoding2)  # True False
        results = np.array(results).tolist()
        return {"success": True,
                "code": 200,
                "msg": f"返回比较结果 [True] or [False]",
                "data": {"compare_result": results}
                }
    except Exception as e:
        return {"success": False,
                "code": 500,
                "msg": str(e),
                "data": None}


@app.post("/faceapi/v1/recognizeCompareEncoding")
async def upload_encoding_list(encodings: List[List[float]]=[[123.1, 123.1], [456.1, 456.2]], encoding: List[float]=[123.2, 123.1], tolerance: float = 0.5):
    """
    人脸比较 返回比较结果 True False
    :param encoding_list: [encoding1, encoding2, encoding3, ...]
    :param encoding: [ 0.00212635  0.18151696 ...  0.08942952 -0.02890663]
    :param show: 是否打印比较结果
    :return: 返回比较结果
    """
    try:
        encodings = np.array(encodings, dtype=np.float64)
        encoding = np.array(encoding, dtype=np.float64)
        face_distances = face_recognition.face_distance(encodings, encoding)
        face_distances = np.array(face_distances < tolerance).tolist()
        return {"success": True,
                "code": 200,
                "msg": f"返回比较结果 [True] or [False]",
                "data": {"compare_result": face_distances}
                }
    except Exception as e:
        return {"success": False,
                "code": 500,
                "msg": str(e),
                "data": None}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
