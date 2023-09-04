import tensorflow as tf
from matplotlib import pyplot as plt
import random
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import os
import shutil
from pathlib import Path

# Collecting data


def train_to_test(train_dir, test_dir, images_for_test_dir=37):
    image_files = [
        f
        for f in os.listdir(train_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]
    selected_images = random.sample(image_files, images_for_test_dir)

    for image in selected_images:
        source_path = os.path.join(train_dir, image)
        dest_path = os.path.join(test_dir, image)
        shutil.move(source_path, dest_path)
    print(f"{images_for_test_dir} random images moved to test_dir")


def make_dir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        print("folder created", dir_name)
        return Path(dir_name)
    else:
        print(f"{folder_name} already exists")


def video_to_frame(train_dir, test_dir, images_to_capture=150):
    label = input("Enter object's Label \n")
    class_labels.append(label)

    label_train_dir = train_dir / label
    label_test_dir = test_dir / label
    label_train_dir.mkdir()
    label_test_dir.mkdir()

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    no_of_images_captured = 0
    images_batch_size = int(images_to_capture / 5)
    while no_of_images_captured < images_to_capture:
        if ret:
            frame_number = 0
            w = int(cap.get(3))
            h = int(cap.get(4))

            for n in [3, 2, 1]:
                copy = frame.copy()

                centre = (int(w / 2), int(h / 2))
                cv2.putText(
                    copy,
                    f"{label}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    1,
                )
                cv2.putText(
                    copy,
                    f"no. of images:{no_of_images_captured+1}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    1,
                )
                cv2.putText(
                    copy, f"{n}", centre, cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3
                )
                cv2.imshow(f"{label}", copy)
                key = cv2.waitKey(1000)

            while frame_number < images_batch_size:
                ret, frame = cap.read()
                if ret:
                    copy = frame.copy()
                    cv2.putText(
                        copy,
                        f"{label}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        1,
                    )
                    cv2.putText(
                        copy,
                        f"no. of images:{no_of_images_captured+1}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        1,
                    )

                    frame_number += 1
                    no_of_images_captured += 1
                    frame_filename = os.path.join(
                        label_train_dir, f"{label}_{no_of_images_captured:04d}.jpg"
                    )
                    cv2.imwrite(str(frame_filename), frame)
                    print(f"Saved {frame_filename}")
                    cv2.imshow(f"{label}", copy)
                    key = cv2.waitKey(1)
                    if key == 27:
                        break

    cap.release()
    cv2.destroyAllWindows()

    return label_train_dir, label_test_dir


folder_name = input("Give a folder name \n")
class_labels = []
data_dir = make_dir(folder_name)
train_dir = data_dir / "train"
test_dir = data_dir / "test"
train_dir.mkdir()
print("train folder created")
test_dir.mkdir()
print("test folder created")

images_to_capture = int(input("How many images do you want to capture? \n"))
images_for_test_dir = int(images_to_capture * 0.25)


label1_train_dir, label1_test_dir = video_to_frame(
    train_dir, test_dir, images_to_capture
)
train_to_test(label1_train_dir, label1_test_dir, images_for_test_dir)

label2_train_dir, label2_test_dir = video_to_frame(
    train_dir, test_dir, images_to_capture
)
train_to_test(label2_train_dir, label2_test_dir, images_for_test_dir)

print(class_labels)

# Model Training


def plot_loss_curves(history):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]

    epochs = range(len(history.history["loss"]))
    plt.plot(epochs, loss, label="training_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.title("loss")
    plt.xlabel("epochs")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(epochs, accuracy, label="training_accuracy")
    plt.plot(epochs, val_accuracy, label="val_accuracy")
    plt.title("accuracy")
    plt.xlabel("epochs")
    plt.legend()
    plt.show()


def imshow(title="Image", image=None, size=8):
    w, h = image.shape[:2]
    aspect_ratio = w / h
    plt.figure(figsize=[aspect_ratio * size, size])
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


def show_random_image(train_data_augmented, no_of_random_images):
    while no_of_random_images == 0:
        no_of_random_images -= 1
        images, labels = train_data_augmented.next()
        random_number = random.randint(0, 32)
        imshow(f"Image Number: {random_number}", images[random_number])


def create_and_train_model(train_data_augmented, test_data):
    model = Sequential(
        [
            Conv2D(
                filters=10, kernel_size=3, activation="relu", input_shape=(224, 224, 3)
            ),
            Conv2D(10, 3, activation="relu"),
            MaxPool2D(pool_size=2),
            Conv2D(10, 3, activation="relu"),
            Conv2D(10, 3, activation="relu"),
            MaxPool2D(),
            Flatten(),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(loss="binary_crossentropy", optimizer=Adam(), metrics=["accuracy"])

    history_1 = model.fit(
        train_data_augmented,
        epochs=5,
        steps_per_epoch=len(train_data_augmented),
        validation_data=test_data,
        validation_steps=len(test_data),
    )
    model.save(f"{folder_name}/{folder_name}_model")
    return history_1


augmented_or_not = "m"
while augmented_or_not.lower() != "y" and augmented_or_not.lower() != "n":
    augmented_or_not = input("Do you want your data to be augmented or not? Y/n \n")

    if augmented_or_not.lower() == "y":
        train_datagen_augmented = ImageDataGenerator(
            rescale=1 / 255.0,
            rotation_range=0.2,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.3,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=True,
        )
        print("data augmented")
    elif augmented_or_not.lower() == "n":
        train_datagen_augmented = ImageDataGenerator(
            rescale=1 / 255.0,
        )
        print("data not augmented")


test_datagen = ImageDataGenerator(rescale=1 / 255.0)

train_data_augmented = train_datagen_augmented.flow_from_directory(
    train_dir, target_size=(224, 224), batch_size=32, class_mode="binary", shuffle=True
)
test_data = test_datagen.flow_from_directory(
    test_dir, target_size=(224, 224), batch_size=32, class_mode="binary", shuffle=True
)
show_random_image(train_data_augmented, no_of_random_images=5)

history = create_and_train_model(train_data_augmented, test_data)

plot_loss_curves(history)

# test model


def predict_model(frame, model_1):
    img = tf.image.resize(frame, (224, 224))
    img = img / 255.0
    img = tf.expand_dims(img, axis=0)
    label = model_1.predict(img)
    print(f"Label: {label}")
    return label


def get_label_and_prob(frame, model_1):
    label = predict_model(frame, model_1)

    if label[0][0] > 0.5:
        class_label = class_labels[1]
        prob1 = int((label[0][0]) * 100)
        prob2 = int(100 - prob1)
        return class_label, prob1, prob2
    else:
        class_label = class_labels[0]
        prob2 = int((1 - label[0][0]) * 100)
        prob1 = int(100 - prob2)
        return class_label, prob1, prob2


cap = cv2.VideoCapture(0)
model = load_model(f"{folder_name}/{folder_name}_model")
while True:
    ret, frame = cap.read()
    if ret:
        copy = frame.copy()
        label, prob1, prob2 = get_label_and_prob(copy, model)
        cv2.putText(
            copy,
            f"Class Label:  {label}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            copy,
            f"Probablity of {class_labels[0]} :{prob2}%",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            copy,
            f"Probablity of {class_labels[1]} :{prob1}%",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.imshow("img", copy)
        key = cv2.waitKey(1)
        if key == 27:
            break
cap.release()
cv2.destroyAllWindows


# Proggramed By Samyak A. Nahar


# Libraries Used = cv2,Tensorflow,random,matplotlib,os,shutil and Pathlib
