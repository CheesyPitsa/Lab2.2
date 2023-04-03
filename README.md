# Лабораторная работа 2, команда Aboba
Модель обучена распознавать, что изображено на фото.

Для обучения и тестирования использован набор данных cifar10, содержащий фото животных и транспорта.

```python
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
```

Слои, дополнительно добавленные в модель:

```python
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
```

Параметры компиляции модели:

```python
model.compile(optimizer = tf.keras.optimizers.Nadam(use_ema=True),
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
```

Результаты обучения модели:

```
Test loss: 1.3398584127426147

Test accuracy: 0.7263000011444092
```
![image](https://user-images.githubusercontent.com/113666100/229601187-584b7ea4-a3b3-40a4-a0f7-2f3e961e1479.png)
![image](https://user-images.githubusercontent.com/113666100/229601339-3414661a-6d10-4c6f-abc4-6a7118c89dc4.png)

![image](https://user-images.githubusercontent.com/113666100/229601459-17a9b192-0076-45d6-95c1-3fe3b3de91e9.png)
![image](https://user-images.githubusercontent.com/113666100/229601538-15cf97a8-0713-41df-8d41-0179abf6f866.png)




