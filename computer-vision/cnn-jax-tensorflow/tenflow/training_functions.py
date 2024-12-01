# TensorFlow and other libraries
import tensorflow as tf
import numpy as np
import os


def train_step(
    model: tf.keras.Model,
    images: tf.Tensor,
    labels: tf.Tensor,
    loss_object: tf.keras.losses.Loss,
    optimizer: tf.keras.optimizers.Optimizer,
    train_loss: tf.keras.metrics.Mean,
    train_accuracy: tf.keras.metrics.SparseCategoricalAccuracy,
):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


def val_step(
    model: tf.keras.Model,
    images: tf.Tensor,
    labels: tf.Tensor,
    loss_object: tf.keras.losses.Loss,
    val_loss: tf.keras.metrics.Mean,
    val_accuracy: tf.keras.metrics.SparseCategoricalAccuracy,
):
    predictions = model(images, training=False)
    loss = loss_object(labels, predictions)

    val_loss(loss)
    val_accuracy(labels, predictions)


def test_step(
    model: tf.keras.Model,
    images: tf.Tensor,
    labels: tf.Tensor,
    loss_object: tf.keras.losses.Loss,
    test_loss: tf.keras.metrics.Mean,
    test_accuracy: tf.keras.metrics.SparseCategoricalAccuracy,
):
    predictions = model(images, training=False)
    loss = loss_object(labels, predictions)

    test_loss(loss)
    test_accuracy(labels, predictions)
    return test_loss.result(), test_accuracy.result()
