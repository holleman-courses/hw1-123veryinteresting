
# TensorFlow and tf.keras


import tensorflow as tf

import tensorflow.keras
import keras
from keras import layers, models, Input, Sequential


from keras.losses import SparseCategoricalCrossentropy
#from sklearn.model_selection import train_test_split

# Helper libraries
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image

import random as rand

import numpy as np

import os
import sys
import time
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


global NUM_OF_EPOCHS
NUM_OF_EPOCHS = 30


print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")


input_shape=(32, 32, 3)

def build_model1():
  model = tf.keras.Sequential([
    Input(shape=input_shape),
    layers.Flatten(),
    layers.Dense(128, activation='leaky_relu'),
    layers.Dense(128, activation='leaky_relu'),
    layers.Dense(128, activation='leaky_relu'),    
    layers.Dense(10)
  ])
  return model

def build_model2():
  model = tf.keras.Sequential([
    Input(shape=input_shape),
    layers.Conv2D(32, kernel_size=(3,3), strides=(2,2), 
                  activation="relu", padding='same'),
    layers.BatchNormalization(),
    
    layers.Conv2D(64, kernel_size=(3,3), strides=(2,2),
                  activation="relu", padding='same'),
    layers.BatchNormalization(), 
    
    layers.Conv2D(128, kernel_size=(3,3), activation="relu", padding='same'),
    layers.BatchNormalization(), 
    
    layers.Conv2D(128, kernel_size=(3,3), activation="relu", padding='same'),
    layers.BatchNormalization(), 
    
    layers.Conv2D(128, kernel_size=(3,3), activation="relu", padding='same'),
    layers.BatchNormalization(), 
    
    layers.Conv2D(128, kernel_size=(3,3), activation="relu", padding='same'),
    layers.BatchNormalization(), 
    
    layers.Flatten(),
    layers.Dense(10)
  ])
  return model

def build_model3():
  model = tf.keras.Sequential([
    Input(shape=input_shape),
    layers.SeparableConv2D(32, kernel_size=(3,3), strides=(2,2), 
                  activation="relu", padding='same'),
    layers.BatchNormalization(),
    
    layers.SeparableConv2D(64, kernel_size=(3,3), strides=(2,2),
                  activation="relu", padding='same'),
    layers.BatchNormalization(), 
    
    layers.SeparableConv2D(128, kernel_size=(3,3), activation="relu", padding='same'),
    layers.BatchNormalization(), 
    
    layers.SeparableConv2D(128, kernel_size=(3,3), activation="relu", padding='same'),
    layers.BatchNormalization(), 
    
    layers.SeparableConv2D(128, kernel_size=(3,3), activation="relu", padding='same'),
    layers.BatchNormalization(), 
    
    layers.SeparableConv2D(128, kernel_size=(3,3), activation="relu", padding='same'),
    layers.BatchNormalization(), 
    
    layers.Flatten(),
    layers.Dense(10)
  ])
  return model
    
    
def build_model50k():
  model = tf.keras.Sequential([
    Input(shape=input_shape),
    layers.SeparableConv2D(32, kernel_size=(3,3), strides=(2,2),  padding='same'),
    layers.ReLU(),
    layers.BatchNormalization(),
    
    layers.SeparableConv2D(64, kernel_size=(3,3), strides=(2,2), padding='same'),
    layers.ReLU(),
    layers.BatchNormalization(), 
    
    layers.SeparableConv2D(128, kernel_size=(3,3), padding='same'),
    layers.ReLU(),
    layers.BatchNormalization(), 
    
    layers.SeparableConv2D(128, kernel_size=(3,3), padding='same'),
    layers.ReLU(),
    layers.BatchNormalization(), 

    layers.GlobalMaxPool2D(),

    layers.Flatten(),

    layers.Dense(64),    
    layers.ReLU(),

    layers.Dense(32),    
    layers.ReLU(),

    layers.Dense(10),

  ])
  return model





def train_model(model, train_images, train_labels, val_images, val_labels, test_images, test_labels, save_path, epochs=10,  csv_path="model_comparison.csv", show_stats=True):


    t0 = time.time()
    history = model.fit(
        train_images,
        train_labels,
        validation_data=(val_images, val_labels),
        epochs=epochs,
        verbose=1
    )
    t1 = time.time()

    total_time = t1 - t0
    time_per_epoch = total_time / max(int(epochs), 1)

    train_eval = model.evaluate(train_images, train_labels, verbose=0)
    val_eval = model.evaluate(val_images, val_labels, verbose=0)
    test_eval = model.evaluate(test_images, test_labels, verbose=0)

    if isinstance(train_eval, (list, tuple)) and len(train_eval) >= 2:
        training_acc = float(train_eval[1])
        training_loss = float(train_eval[0])
    else:
        training_acc = float("nan")
        training_loss = float(train_eval) if train_eval is not None else float("nan")

    if isinstance(val_eval, (list, tuple)) and len(val_eval) >= 2:
        valid_acc = float(val_eval[1])
        valid_loss = float(val_eval[0])
    else:
        valid_acc = float("nan")
        valid_loss = float(val_eval) if val_eval is not None else float("nan")

    if isinstance(test_eval, (list, tuple)) and len(test_eval) >= 2:
        test_acc = float(test_eval[1])
        test_loss = float(test_eval[0])
    else:
        test_acc = float("nan")
        test_loss = float(test_eval) if test_eval is not None else float("nan")

    overfitting_metric = training_acc - valid_acc

    hist = history.history

    train_acc_series = np.asarray(hist["accuracy"], dtype=float)
    val_acc_series = np.asarray(hist["val_accuracy"], dtype=float)

    train_loss_series = np.asarray(hist["loss"], dtype=float)
    val_loss_series = np.asarray(hist["val_loss"], dtype=float)


    epochs_until = {}
    for pct in range(10, 100, 10):
        epochs_until[pct] = ""

    if val_acc_series is not None and val_acc_series.size:
        for pct in range(10, 100, 10):
            thr = pct / 100.0
            idx = np.where(val_acc_series >= thr)[0]
            epochs_until[pct] = int(idx[0] + 1) if idx.size else ""

    epoch_recommendation = "unknown"
    if val_acc_series is not None and val_acc_series.size >= 3:
        best_epoch = int(np.argmax(val_acc_series)) + 1
        best_val = float(val_acc_series[best_epoch - 1])
        final_val = float(val_acc_series[-1])
        drop = best_val - final_val
        recent_trend = float(val_acc_series[-1] - val_acc_series[-3])

        if best_epoch < int(epochs) and drop > 0.01:
            epoch_recommendation = "shorter"
        else:
            if recent_trend > 0.002:
                epoch_recommendation = "longer"
            else:
                epoch_recommendation = "about_same"

        if overfitting_metric > 0.05 and epoch_recommendation == "longer":
            epoch_recommendation = "about_same"

    row = {
        "model_name": save_path,
        "training_acc": training_acc,
        "valid_acc": valid_acc,
        "test_acc": test_acc,
        "overfitting_metric": overfitting_metric,
        "time_per_epoch": time_per_epoch,
        "epoch_recommendation": epoch_recommendation
    }
    for pct in range(10, 100, 10):
        row[f"epochs_until_{pct}"] = epochs_until[pct]

    fieldnames = [
        "model_name",
        "training_acc",
        "valid_acc",
        "test_acc",
        "overfitting_metric",
        "time_per_epoch",
        "epoch_recommendation",
    ] + [f"epochs_until_{pct}" for pct in range(10, 100, 10)]

    need_header = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if need_header:
            writer.writeheader()
        writer.writerow(row)

    model.save(save_path)

    plots_dir = os.path.join(os.getcwd(), "training_plots")
    os.makedirs(plots_dir, exist_ok=True)

    safe_name = "".join([c if c.isalnum() or c in ("-", "_") else "_" for c in save_path])
    plot_path = os.path.join(plots_dir, f"{safe_name}_learning_curves.png")

    epoch_axis = None
    n_epochs_plot = None
    for series in (train_acc_series, val_acc_series, train_loss_series, val_loss_series):
        if series is not None:
            n_epochs_plot = int(series.size)
            break
    if n_epochs_plot is not None and n_epochs_plot > 0:
        epoch_axis = np.arange(1, n_epochs_plot + 1)

        fig, axs = plt.subplots(2, 1, figsize=(8, 8))

        if train_acc_series is not None:
            axs[0].plot(epoch_axis, train_acc_series, label="train_acc")
        if val_acc_series is not None:
            axs[0].plot(epoch_axis, val_acc_series, label="val_acc")
        axs[0].set_xlabel("epoch")
        axs[0].set_ylabel("accuracy")
        axs[0].set_title(f"{save_path}: accuracy")
        axs[0].legend()

        if train_loss_series is not None:
            axs[1].plot(epoch_axis, train_loss_series, label="train_loss")
        if val_loss_series is not None:
            axs[1].plot(epoch_axis, val_loss_series, label="val_loss")
        axs[1].set_xlabel("epoch")
        axs[1].set_ylabel("loss")
        axs[1].set_title(f"{save_path}: loss")
        axs[1].legend()

        fig.tight_layout()
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)

    if show_stats:
        print(f"Appended results to: {csv_path}")
        print(f"Saved model to: {save_path}")
        if n_epochs_plot is not None and n_epochs_plot > 0:
            print(f"Saved plot to: {plot_path}")
        print(row)

    return history



def split_train_valid(train_images_full, train_labels_full, valid_size):
    assert 0 < valid_size < 0.5

    train_images_full = np.asarray(train_images_full)
    train_labels_full = np.asarray(train_labels_full)

    arr_len = train_images_full.shape[0]
    assert train_labels_full.shape[0] == arr_len

    val_target = int(valid_size * arr_len)
    train_target = arr_len - val_target

    print(val_target, train_target)

    img_shape = train_images_full.shape[1:]

    train_images = np.empty((train_target, *img_shape), dtype=train_images_full.dtype)
    val_images = np.empty((val_target, *img_shape), dtype=train_images_full.dtype)

    train_labels = np.empty((train_target, *train_labels_full.shape[1:]), dtype=train_labels_full.dtype)
    val_labels = np.empty((val_target, *train_labels_full.shape[1:]), dtype=train_labels_full.dtype)


    

    train_i = 0
    val_i = 0

    for k in range(arr_len):
        if train_i >= train_target:
            val_images[val_i] = train_images_full[k]
            val_labels[val_i] = train_labels_full[k]
            val_i += 1
            continue

        if val_i >= val_target:
            train_images[train_i] = train_images_full[k]
            train_labels[train_i] = train_labels_full[k]
            train_i += 1
            continue

        if rand.random() < valid_size:
            val_images[val_i] = train_images_full[k]
            val_labels[val_i] = train_labels_full[k]
            val_i += 1
        else:
            train_images[train_i] = train_images_full[k]
            train_labels[train_i] = train_labels_full[k]
            train_i += 1

    return train_images, val_images, train_labels, val_labels



# no training or dataset construction should happen above this line
# also, be careful not to unindent below here, or the code be executed on import
if __name__ == '__main__':


    train_model_1 = False
    train_model_2 = False
    train_model_3 = False

    train_model_50k = False

    test_model_1 = False
    test_model_2 = False
    test_model_3 = False

    test_model_50k = False

    if '--train_model_1' in sys.argv:
        train_model_1 = True

    if '--train_model_2' in sys.argv:
        train_model_2 = True

    if '--train_model_3' in sys.argv:
        train_model_3 = True

    if '--train_model_50k' in sys.argv:
        train_model_50k = True

    if '--test_model_1' in sys.argv:
        test_model_1 = True

    if '--test_model_2' in sys.argv:
        test_model_2 = True

    if '--test_model_3' in sys.argv:
        test_model_3 = True

    if '--test_model_50k' in sys.argv:
        test_model_50k = True

    


  ########################################
  ## Add code here to Load the CIFAR10 data set

    (train_images_full, train_labels_full), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

    class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

    train_images_full = train_images_full.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0

    train_images, val_images, train_labels, val_labels = split_train_valid(
        train_images_full,
        train_labels_full,
        valid_size=0.3,
    )

    


    model1 = build_model1()
    model2 = build_model2()
    model3 = build_model3()

    model50k = build_model50k()
    models = [model1, model2, model3, model50k]



    trains = [train_model_1, train_model_2, train_model_3, train_model_50k]

    tests = [test_model_1, test_model_2, test_model_3, test_model_50k]



    for k in range(0, len(models)):

        model = models[k]

        model.compile(optimizer='adam',
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])

        model.summary()



        if trains[k]:

        
            train_model(
                model,
                train_images,
                train_labels,
                val_images,
                val_labels,
                test_images,
                test_labels,
                save_path=f'./model_{k+1}',
                epochs=NUM_OF_EPOCHS
            )

            model.save(f'model_{k+1}.h5')



        if tests[k]:

            if not trains[k]:

                try:
                    model.load_weights(f'./model_{k+1}.h5')
                except e:
                    print(e)



            test_img = np.array(keras.utils.load_img('./test_image.png',grayscale=False,color_mode='rgb',target_size=(32,32)))

            class_vector = model1.predict(truck_img)
            label_idx = np.argmax(class_vector)

            predicted_class_label = class_names[label_idx]

            true_class_label = 'dog'

            print(f'\nThe test image belongs to class {true_class_label}.')
            print(f'Model_{k+1} predicted a class of {predicted_class_label}.')

            if true_class_label == predicted_class_label:
                print(f'Model_{k+1} correctly classified the test image.\n')

            else:
                print(f'Model_{k+1} incorrectly classified the test image.\n')



    model50k.summary()
    model50k.save("best_model.h5")

