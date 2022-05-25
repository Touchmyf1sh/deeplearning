from keras.datasets import mnist  # 导入数据集
from keras import models  # 创建神经网络及其启动
from keras import layers  # 创建中间层
from keras.utils import to_categorical  # 准备标签

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()  # 导入数据集和测试集
network = models.Sequential()  # 创建神经网络
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))  # 创建一个中间层，激活函数为relu
network.add(layers.Dense(10, activation='softmax'))  # 创建中间层，激活函数为softmax
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])  # 编译神经网络 并确定损失函数（loss），优化器（optimizer），和监控的指标（metrics）
train_images = train_images.reshape((60000, 28 * 28)).astype('float32') / 255  # 将图像变换尺寸使符合函数
test_images = test_images.reshape((10000, 28 * 28)).astype('float32') / 255
train_labels = to_categorical(train_labels)  # 准备标签
test_labels = to_categorical(test_labels)
network.fit(train_images, train_labels, epochs=5, batch_size=128)  # 开始数据拟合
