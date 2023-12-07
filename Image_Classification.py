import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm

image_dir=r"E:\Sem 3\DL\Image Classification\cropped"
messi_images=os.listdir(image_dir+ '\\lionel_messi')
maria_images=os.listdir(image_dir+ '\\maria_sharapova')
roger_images=os.listdir(image_dir+ '\\roger_federer')
serena_images=os.listdir(image_dir+ '\\serena_williams')
virat_images=os.listdir(image_dir+ '\\virat_kohli')
dataset=[]
label=[]
img_siz=(128,128)


for i , image_name in tqdm(enumerate(messi_images),desc="lionel_messi"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/lionel_messi/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(0)
        
        
for i ,image_name in tqdm(enumerate(maria_images),desc="maria_sharapova"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/maria_sharapova/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(1)

for i ,image_name in tqdm(enumerate(roger_images),desc="roger_federer"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/roger_federer/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(2)        

for i ,image_name in tqdm(enumerate(serena_images),desc="serena_williams"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/serena_williams/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(3)

for i ,image_name in tqdm(enumerate(virat_images),desc="virat_kohli"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/virat_kohli/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(4)

        
dataset=np.array(dataset)
label = np.array(label)

print("--------------------------------------\n")
print('Dataset Length: ',len(dataset))
print('Label Length: ',len(label))

print("--------------------------------------\n")
print("Train-Test Split")
x_train,x_test,y_train,y_test=train_test_split(dataset,label,test_size=0.2,random_state=42)
print("--------------------------------------\n")

print("Normalaising the Dataset. \n")

x_train = x_train.astype('float')/255
x_test = x_test.astype('float')/255

print("--------------------------------------\n")


model=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(128,128,3)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256,activation='relu'),
    tf.keras.layers.Dropout(.5),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(5,activation='softmax')
])
print("--------------------------------------\n")
model.summary()
print("--------------------------------------\n")

model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])


print("--------------------------------------\n")
print("Training Started.\n")
history=model.fit(x_train,y_train,epochs=50,batch_size =32,validation_split=0.3)
print("Training Finished.\n")


print("--------------------------------------\n")
print("Model Evalutaion Phase.\n")
loss,accuracy=model.evaluate(x_test,y_test)
print(f'Accuracy: {round(accuracy*100,2)}')
print("--------------------------------------\n")
y_pred=model.predict(x_test)
y_pred = np.argmax(y_pred,axis=1)
print('classification Report\n',classification_report(y_test,y_pred))
print("--------------------------------------\n")


print("--------------------------------------\n")
print("Model Prediction.\n")

def preprocess_single_image(image_path):
    img_size = (128, 128)
    image = cv2.imread(image_path)
    image = Image.fromarray(image, 'RGB')
    image = image.resize(img_size)
    image = np.array(image)
    image = image.astype('float32') / 255.0
    return image

# List of paths to multiple images you want to predict
image_paths_to_predict = [
    r'cropped\lionel_messi\lionel_messi2.png',
    r'cropped\maria_sharapova\maria_sharapova5.png',
    r'cropped\roger_federer\roger_federer2.png',
    r'cropped\serena_williams\serena_williams2.png',
    r'cropped\virat_kohli\virat_kohli2.png'
]
# Preprocess and predict for each image
for image_path in image_paths_to_predict:
    single_image = preprocess_single_image(image_path)
    single_image = np.expand_dims(single_image, axis=0)
    predictions = model.predict(single_image)
    predicted_class = np.argmax(predictions)
    
    class_names = ['lionel Messi', 'Maria Sharapova', 'Roger Federer', 'serena williams', 'Virat kohli']
    predicted_label = class_names[predicted_class]

    print(f" Predicted image: {predicted_label}")