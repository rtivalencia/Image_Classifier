import os 
import numpy as np
from PIL import Image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Reshape, LeakyReLU

from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split


X_train = []
y_train = []
label_encoder = LabelEncoder()
counter = 0

for root, dirs, files in os.walk("training_set"):
    for filename in files:
        counter += 1
        if filename.endswith(".jpg"):                       # root contains path to a directory
            img_path = os.path.join(root, filename)         # os.path.join to concatenate the dir path and the image file name (used to gain access to the image file)
            img = Image.open(img_path).convert('RGB')       # opens the img with the path 
            img = img.resize((64, 64))                      # resizing the image to fit the model
            img_array = np.array(img)                       # numpy converts the open image file into an NumPy array (pixel values)
            if img_array.shape == (64, 64, 3):
                X_train.append(img_array)
                label = root.split("\\")[-1]                # get the bird species from the folder name 
                y_train.append(label)
        print(counter)


# Convert labels to categorical
y_train = label_encoder.fit_transform(y_train)
y_train = to_categorical(y_train, num_classes=None)



# Reshape and normalize data
X_train = np.array(X_train)                                                 # converts list into a NumPy array
X_train = np.reshape(X_train, (X_train.shape[0], 64, 64, 3))                # reshapes the array to have dimensions (# of samples, width, height, RGB)
X_train = X_train.astype('float32') / 255.0                                 # normalizes the pixel value between 0 & 1 

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

# Reshape and normalize data
X_val = np.array(X_val)
X_val = np.reshape(X_val, (X_val.shape[0], 64, 64, 3))
X_val = X_val.astype('float32') / 255.0


print(X_train.shape)
print(y_train.shape)

print("Number of training samples:", len(X_train))
print("Number of training labels:", len(y_train))
print("Number of validation samples:", len(X_val))
print("Number of validation labels:", len(y_val))

# Define CNN model architecture
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(8 * 8 * 64, activation='relu'))
model.add(Dense(25, activation='softmax')) # 25 refers to the number of different classes wanting to be trained 

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Train the model
model.fit(X_train, y_train, batch_size=1, validation_data=(X_val, y_val))

# Evaluate the model on the validation data
score = model.evaluate(X_val, y_val, verbose=0)
print('Validation loss:', score[0])
print('Validation accuracy:', score[1])
model.summary()  # prints the model summary

#--------------------------------------------------------- flow_from_directory variant of above code ---------------------------------------------------------
import os 
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Reshape, LeakyReLU

from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split


# Define data generator for training and validation sets
# Make sure data is presplit into validation_set and training_set directories
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

val_generator = val_datagen.flow_from_directory(
        'validation_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

# Define CNN model architecture
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(8 * 8 * 64, activation='relu'))
model.add(Dense(25, activation='softmax')) # 25 refers to the number of different classes wanting to be trained 

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=10, validation_data=val_generator)

# Evaluate the model on the validation data
score = model.evaluate(val_generator, verbose=0)
print('Validation loss:', score[0])
print('Validation accuracy:', score[1])
model.summary()
