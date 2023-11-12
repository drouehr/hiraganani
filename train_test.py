import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

dir = './hiragana'
images = []
labels = []

for filename in os.listdir(dir):
  if filename.endswith('.png'):
    img = cv2.imread(os.path.join(dir, filename))
    # convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (127, 128))
    # normalize
    img = img / 255.0
    # add image and label to lists
    images.append(img)
    # label from filename
    label = int(filename.split('_')[0])
    labels.append(label)

X = np.array(images)
y = np.array(labels)
# fit to keras model input format
X = X.reshape(-1, 127, 128, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

datagen = ImageDataGenerator(
  rotation_range=6.5,
  width_shift_range=0.05,
  height_shift_range=0.05,
  shear_range=5,
  fill_mode='nearest'
)
datagen.fit(X_train)

# integers to binary class matrices
y_train = to_categorical(y_train, num_classes=46)
y_test = to_categorical(y_test, num_classes=46)

# model definition
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(127, 128, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(46, activation='softmax'))
# 46 characters

# compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# train model
# no datagen: model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=1)
history = model.fit(datagen.flow(X_train, y_train, batch_size=128), steps_per_epoch=len(X_train)/128, epochs=22, validation_data=(X_test, y_test), verbose=1)

# save
timestamp = time.strftime('%Y%m%d_%H%M%S')
model.save('./models/model_'+timestamp)

report_dir = './reports/report_' + timestamp
if not os.path.exists(report_dir):
  os.makedirs(report_dir)

# feature mapping on a random test image
fmap_idxs = [0, 2, 4]
outputs = [model.layers[i].output for i in fmap_idxs]
mod = Model(inputs=model.inputs, outputs=outputs)
random_index = np.random.randint(0, len(X_test))
img = X_test[random_index].reshape(1, 127, 128, 1)
feature_maps = mod.predict(img)
for layer_num, fmap in enumerate(feature_maps):
  ix = 1
  square = int(np.sqrt(fmap.shape[-1]))
  fig, ax = plt.subplots(square, square, figsize=(10,10))
  for _ in range(square):
    for _ in range(square):
      # specify subplot and turn off axis
      ax = plt.subplot(square, square, ix)
      ax.set_xticks([])
      ax.set_yticks([])
      # plot filter channel in grayscale
      if ix-1 < fmap.shape[-1]:
        plt.imshow(fmap[0, :, :, ix-1], cmap='gray')
      ix += 1
  plt.savefig(report_dir + '/feature_map_layer_' + str(layer_num) + '.png')
  plt.close()

# plot loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.title('loss plot')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.savefig(report_dir + '/loss_plot.png')
plt.close()

# plot accuracy
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.title('accuracy plot')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.savefig(report_dir + '/accuracy_plot.png')
plt.close()

# confusion matrix
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize = (10,7))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('predicted')
plt.ylabel('truth')
plt.savefig(report_dir + '/confusion_matrix.png')
plt.close()

# classification report
class_report = classification_report(y_true, y_pred_classes)

with open(report_dir + '/classification_report.txt', 'w') as f:
  f.write(class_report)

# evaluate
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
print('\ntest accuracy = ', test_acc)
