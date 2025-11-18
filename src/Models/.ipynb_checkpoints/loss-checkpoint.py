import tensorflow as tf

class SoftF1Loss(tf.keras.losses.Loss):
    def __init__(self, num_classes, epsilon=1e-6, name="soft_f1_loss"):
        super().__init__(name=name)
        self.num_classes = num_classes
        self.epsilon = epsilon

    def call(self, y_true, y_pred):
        # y_true: [B] integer class ids
        # y_pred: [B, C] logits or probabilities

        y_pred = tf.nn.softmax(y_pred, axis=-1)

        # one-hot target
        y_true_oh = tf.one_hot(tf.cast(tf.squeeze(y_true), tf.int32),
                               depth=self.num_classes)


        tp = tf.reduce_sum(y_pred * y_true_oh, axis=0)
        fp = tf.reduce_sum(y_pred * (1.0 - y_true_oh), axis=0)
        fn = tf.reduce_sum((1.0 - y_pred) * y_true_oh, axis=0)

        precision = tp / (tp + fp + self.epsilon)
        recall    = tp / (tp + fn + self.epsilon)

        f1 = 2 * precision * recall / (precision + recall + self.epsilon)
        macro_f1 = tf.reduce_mean(f1)

        return 1.0 - macro_f1   # 0 = tökéletes, 1 = nagyon rossz
