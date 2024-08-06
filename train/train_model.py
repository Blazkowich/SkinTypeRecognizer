# train_model.py
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model

def train_model(train_dir, valid_dir, model_path):
    # Image dimensions
    img_width, img_height = 150, 150

    # Data Augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    valid_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='categorical')

    valid_generator = valid_datagen.flow_from_directory(
        valid_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='categorical')

    # Build the model
    inputs = Input(shape=(img_width, img_height, 3))
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(3, activation='softmax')(x)  # 3 classes: dry, normal, oily

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(train_generator, epochs=10, validation_data=valid_generator)

    # Save the model
    model.save(model_path)

if __name__ == '__main__':
    # Define paths
    train_dir = r'D:\Nugzar\Oily-Dry-Skin-Types\train'
    valid_dir = r'D:\Nugzar\Oily-Dry-Skin-Types\valid'
    model_path = 'skin_type_model.h5'

    train_model(train_dir, valid_dir, model_path)
