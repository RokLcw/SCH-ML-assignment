import os
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import sys

width = 4096
height = 4096

folder_path = "./disasmbled"
folder_lists = os.listdir(folder_path)

maxlen = 0

for folder in folder_lists: 
    file_path = os.path.join(folder_path, folder)
    file_lists = os.listdir(file_path)

    for file in file_lists:
        with open(os.path.join(file_path, file), 'r') as f:
            asm = f.readlines()
            opcode = [code[:-1].split()[0] for code in asm]
        # 가장 긴 opcode 찾기
        opcode = ''.join(opcode)
        if(len(opcode) > maxlen):
            maxlen = len(opcode)
        opcode_list = opcode
        opcode_list = list(opcode_list)
        for length in range(0, len(opcode_list)):
            opcode_list[length] = ord(opcode_list[length])

        image = Image.new('L', (width, height))
        image.putdata(opcode_list)
        image = image.resize((64, 64))
        asm_name = file.split('.asm')[0]
        if 'test' in folder:
            imagename = f"./csv/opcode_image_string/img/test/{folder}/{asm_name}.png"
        if 'train' in folder:
            imagename = f"./csv/opcode_image_string/img/train/{folder}/{asm_name}.png"
        if 'valid' in folder:
            if 'benign' in folder:
                imagename = f"./csv/opcode_image_string/img/train/train__benign/{asm_name}.png"
            if 'malware' in folder:
                imagename = f"./csv/opcode_image_string/img/train/train__malware/{asm_name}.png"
        
        image.save(imagename)

print("전처리 완료")

train_folder_path = "./csv/opcode_image_string/img/train"

# train 데이터 셋 가져오기
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_folder_path,
    image_size=(64,64),
    batch_size=64,
    subset='training',
    validation_split=0.2,
    color_mode = 'grayscale',
    seed=1234
)

# validation 데이터 셋 가져오기
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_folder_path,
    image_size=(64,64),
    batch_size=64,
    subset='validation',   
    color_mode = 'grayscale',
    validation_split=0.2,
    seed=1234
)

# def Pretreatment(i, result):
#     i=tf.cast(i/255.0, tf.float32)
#     return i, result

# train_ds = train_ds.map(Pretreatment)
# val_ds = val_ds.map(Pretreatment)


# for i, result in train_ds.take(1):
#     print(i)

model = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal', input_shape=(64, 64, 1)),   # 이미지 증강
    # tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),   # 이미지 증강
    # tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),   # 이미지 증강
    tf.keras.layers.Conv2D(32, (3,3), padding="same", activation="relu"),  # 컨볼루션 레이어
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3,3), padding="same", activation="relu"),  # 컨볼루션 레이어
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.2),   # 학습용 데이터 외우는거 방지, 오버비팅 완화
    tf.keras.layers.Conv2D(128, (3,3), padding="same", activation="relu"),  # 컨볼루션 레이어
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),  # 1차원 데이터로 만들어줌 (이미지를 한줄로 쭉)
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),   # 학습용 데이터 외우는거 방지, 오버비팅 완화
    tf.keras.layers.Dense(1, activation="sigmoid"), # 클래스 네임(카테코리) 1개
])

model.summary()

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
result_train = model.fit(train_ds, validation_data=val_ds, epochs=280) # validation_data: 오버피팅 방지(학습용 데이터 외움), epochs 마다 테스트

# 그래프로 Validation accuracy 와 Trainning accuracy 비교
# plt.plot(result_train.history['accuracy'], label='Train Accuracy')
# plt.plot(result_train.history['val_accuracy'], label='Validation Accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend()
# plt.show()

# test 데이터 셋 가져오기
folder_path = "./csv/opcode_image_string/img/test"
folder_lists = os.listdir(folder_path)

image_list = []
image_label = []

for folder in folder_lists: 
	file_path = os.path.join(folder_path, folder)
	file_lists = os.listdir(file_path)

	for file in file_lists:
		path = file_path + "\\" + file
		image = Image.open(path)
		image_numpy = np.array(image)
		image_list.append(image_numpy)

		if 'benign' in folder:
			image_label.append(0)
		elif 'malware' in folder:
			image_label.append(1)

testX = pd.DataFrame()
testY = pd.DataFrame()

testX = np.array(image_list)
testY = np.array(image_label)

# 테스트 데이터셋으로 정답률 확인
score = model.evaluate(testX, testY)
print("정답률:", score[1], "\tloss:", score[0])

# 그래프로 train, test accuracy 값 비교
plt.title('model accuracy')
plt.plot(result_train.history['accuracy'], label='train')
plt.plot(result_train.history['val_accuracy'], label='test')
plt.ylabel('model accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()

# 그래프로 train, test loss 값 비교
plt.title('model loss')
plt.plot(result_train.history['loss'], label='train')
plt.plot(result_train.history['val_loss'], label='test')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()