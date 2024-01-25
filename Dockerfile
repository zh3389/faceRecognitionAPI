FROM animcogn/face_recognition:cpu
COPY ./ /app
WORKDIR /app
RUN pip install -r requirements.txt
RUN python /app/utils/faceRecognition.py
CMD [ "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9002", "--reload" ]
