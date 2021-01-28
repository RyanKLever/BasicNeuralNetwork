#!/usr/bin/env python3
import tensorflow as tf
from pathlib import Path
import random
import re
import os
import matplotlib.pyplot as plt


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Disable warning about AVX/FMA

FILE_RECORDS = 60  # There are 60 records per file (years 1959 - 2018)
FILE_COLUMNS = 3  # There are 3 columns per file (Year, Moose, Wolves)
FILES_PER_SERIES = 100  # There are 100 files per series directory
HIDDEN_UNITS = 250
PRINT_EVERY = 100
TRAIN_RATIO = 7 / 8
NUM_EPOCHS = 2000
label_map = {i: series for i, series in enumerate(range(1000, 5000, 500))}


def load_data():
    def load_file(series, run):
        path = Path.cwd() / 'data' / f'series {series}' / f'data_run_{run}.txt'
        with path.open() as f:
            return [
                tuple(map(int, re.split(r'\s+', line.strip())))
                for i, line in enumerate(f) if i > 0
            ]

    def rand_partition(data):
        """ Seperates training data from testing data """
        random.shuffle(data)
        index = int(TRAIN_RATIO * len(data))
        return (data[:index], data[index:])

    train_data = []
    test_data = []
    for label_id, series in label_map.items():
        series_entries = [
            (
                label_id,
                load_file(
                    series,
                    (series + i + 1)
                )
            ) for i in range(FILES_PER_SERIES)
        ]
        train_entries, test_entries = rand_partition(series_entries)
        train_data.extend(train_entries)
        test_data.extend(test_entries)

    random.shuffle(train_data)
    random.shuffle(test_data)
    train_data = [(x,) for x in zip(*train_data)]
    test_data = [(x,) for x in zip(*test_data)]
    return train_data, test_data


def loss(model, x, y, training):
    prediction = model(x, training=training)
    return loss_object(y_true=y, y_pred=prediction)


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


train_data, test_data = load_data()
train_dataset = tf.data.Dataset.from_tensor_slices(
    (
        tf.constant(train_data[1]),
        tf.constant(train_data[0])
    )
)
features, labels = next(iter(train_dataset))
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(FILE_RECORDS, FILE_COLUMNS)),
    tf.keras.layers.Dense(HIDDEN_UNITS, activation=tf.nn.relu),
    tf.keras.layers.Dense(len(label_map))
])
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
l = loss(model, features, labels, training=False)
optimizer = tf.keras.optimizers.RMSprop(
    learning_rate=0.0003,
    rho=0.9,
    momentum=0.0,
    epsilon=1e-07,
    centered=False
)
loss_value, grads = grad(model, features, labels)
optimizer.apply_gradients(zip(grads, model.trainable_variables))

# THIS SECTION IS FOR DEBUGGING
# print(f"Loss test: {l}")
# print(
#     "Step: {}, Initial Loss: {}".format(
#         optimizer.iterations.numpy(),
#         loss_value.numpy()
#     )
# )
# print(
#     "Step: {}, Loss: {}".format(
#         optimizer.iterations.numpy(),
#         loss(model, features, labels, training=True).numpy()
#     )
# )

print("\n\nTraining Phase:")
train_loss_results = []
train_accuracy_results = []
for epoch in range(NUM_EPOCHS):
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    for x, y in train_dataset:
        loss_value, grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        epoch_loss_avg.update_state(loss_value)
        epoch_accuracy.update_state(y, model(x, training=True))
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

    if epoch % PRINT_EVERY == 0:
        print(
            "Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(
                epoch,
                epoch_loss_avg.result(),
                epoch_accuracy.result()
            )
        )

test_dataset = tf.data.Dataset.from_tensor_slices(
    (
        tf.constant(test_data[1]),
        tf.constant(test_data[0])
    )
)
test_accuracy = tf.keras.metrics.Accuracy()
comparisons = []
for (x, y) in test_dataset:
    logits = model(x, training=False)
    prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
    comparisons.append([y.numpy().tolist(), prediction.numpy().tolist()])
    test_accuracy(prediction, y)

print("\n\nTesting Phase:")
print("Test set accuracy: {:.3%}".format(test_accuracy.result()))
print("\nValue  Prediction:")
xs = []
ys = []
for elem in comparisons:
    for (x,y) in zip(elem[0], elem[1]):
        print(f"{x}      {y}")
        xs.append(x)
        ys.append(y)

confusion = tf.math.confusion_matrix(
    labels=xs,
    predictions=ys,
    num_classes=8
)
print(confusion)

# ##Uncomment this section if you'd like to see a graph output
# fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
# fig.suptitle('Training Metrics')
# axes[0].set_ylabel("Loss", fontsize=14)
# axes[0].plot(train_loss_results)
# axes[1].set_ylabel("Accuracy", fontsize=14)
# axes[1].set_xlabel("Epoch", fontsize=14)
# axes[1].plot(train_accuracy_results)
# plt.show()
