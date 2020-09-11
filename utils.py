from os import path

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard


class LRTensorBoard(TensorBoard):
    "Tensorboard that also logs learning rate and settings as a string"

    def __init__(self, log_dir, require_unique_name=True, settings_to_log=None, **kwargs):
        """
        Create tensorboard callback.
        :param log_dir: The experiment name, which is the directory in which the log will be stored.
        :param require_unique_name: Require this name to be unique (good idea as to not overwrite existing runs).
        :param settings_to_log: Log any training paramters here, for reproducibility
        :param kwargs: adittional Tensorboard arguments.
        """
        if require_unique_name and path.exists(log_dir):
            raise ValueError("Tensorboard name must be unique")

        file_writer = tf.summary.create_file_writer(log_dir + "/train")
        with file_writer.as_default():
            tensor = tf.convert_to_tensor(str(settings_to_log))
            tf.summary.text("Run Settings", tensor, step=0)
            file_writer.flush()

        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)




def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
