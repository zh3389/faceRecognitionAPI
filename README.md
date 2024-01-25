## 人脸识别Docker服务
采用Docker + Python + FastAPI的方式开发人脸识别服务

### 依赖

- FastAPI
- uvicorn
- pillow
- numpy
- python-multipart

### 快速开始

环境安装
```shell
git clone 本项目地址
pip install -r requirements.txt
python app.py
# 浏览器访问文档页面
http://localhost:9002/docs
```


### 客户端

```python
import requests

url = 'http://localhost:9002/faceapi/v1/detectBox'
files = {'imgFile': ('file', open('/path/to/file', 'rb'))}

response = requests.get(url, files=files)

print(response.text)

```

```shell
curl --location --request GET 'http://localhost:9002/faceapi/v1/detectBox' \
--form 'imgFile=@"/path/to/file"'
```

### Docker打包运行

```shell
docker build -t face_recognition_service .
docker run -d -p 9002:9002 face_recognition_service:latest
浏览器打开: http://localhost:9002/docs
```