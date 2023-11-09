import os
import cv2
import numpy as np
from keras.models import load_model
from keras.utils import to_categorical

map = ['A','I','U','E','O','KA','KI','KU','KE','KO','SA','SHI','SU','SE','SO','TA','CHI','TSU','TE','TO','NA','NI','NU','NE','NO','HA','HI','FU','HE','HO','MA','MI','MU','ME','MO','YA','YU','YO','RA','RI','RU','RE','RO','WA','WO','N']

# get latest model
model_dir = os.listdir('./models')
# is directory
model_files = [file for file in os.listdir('./models') if os.path.isdir('./models/' + file)]
model_files.sort(reverse=True)
model = load_model('./models/' + model_files[0])
print("loading model: " + model_files[0])

test_dir = './test_images'
test_images = []
test_labels = []
for filename in os.listdir(test_dir):
  if filename.endswith('.png'):
    # same preprocessing as training
    img = cv2.imread(os.path.join(test_dir, filename))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (127, 128))
    img = img / 255.0
    test_images.append(img)
    label = int(filename.split('_')[0])
    test_labels.append(label)

X_test_new = np.array(test_images)
X_test_new = X_test_new.reshape(-1, 127, 128, 1)
y_test_new = np.array(test_labels)

preds = model.predict(X_test_new, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_test_new = to_categorical(y_test_new, num_classes=46)

for i in range(len(y_pred)):
  incorrectIndicator = ''
  if y_pred[i] != test_labels[i]:
    incorrectIndicator = ' <incorrect>'
  print("image " + str(i) + ": predicted = " + str(y_pred[i]) + " (" + map[y_pred[i]] + "), actual = " + str(test_labels[i]) + " (" + map[test_labels[i]] + ")" + incorrectIndicator)

correct_preds = np.sum(y_pred == test_labels)
total_preds = len(y_pred)
accuracy = correct_preds / total_preds
print("accuracy on test images = ", correct_preds, "/", total_preds, ", ", accuracy*100, "%")