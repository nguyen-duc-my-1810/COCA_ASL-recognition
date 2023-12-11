import pickle
import numpy as np

import keras
import keras_tuner
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import tensorflow.keras.backend as K

# Clear the TensorFlow session
K.clear_session()

right_hand_landmarks = pickle.load(open("D:\\[COCA]_ASL_recognition\Model\\right_hand_data.pickle", 'rb'))

right_hand_data = right_hand_landmarks['data']
right_hand_labels = right_hand_landmarks['labels']

left_hand_landmarks = pickle.load(open("D:\\[COCA]_ASL_recognition\Model\\left_hand_data.pickle", 'rb'))

left_hand_data = left_hand_landmarks['data']
left_hand_labels = left_hand_landmarks['labels']

def cut_landmarks(data, labels):
    new_data = []
    new_labels = []

    for landmarks, label in zip(data, labels):
        if len(landmarks) == 126:
            new_landmarks = landmarks[:63] 
            new_data.append(new_landmarks)
            new_labels.append(label)
        else:
            new_data.append(landmarks)
            new_labels.append(label)

    data = new_data
    labels = new_labels

    return data, labels


right_hand_data, right_hand_labels = cut_landmarks(right_hand_data, right_hand_labels)
left_hand_data, left_hand_labels = cut_landmarks(left_hand_data, left_hand_labels)

right_data_train, right_data_val, right_labels_train, right_labels_val = train_test_split(right_hand_data, right_hand_labels, test_size=0.2, shuffle=True, stratify=right_hand_labels)

def preprocessing_training(data, labels):
    data_np = np.array(data)
    labels_np = np.array(labels)
   
    train_sample = len(data_np)
   
    timesteps = 1  # Số lượng thời điểm cho mỗi mẫu (có thể thay đổi tùy theo mô hình của bạn)
    features = len(data_np[0])

    data = data_np.reshape(train_sample, timesteps, features)
    labels = to_categorical(labels_np)
    
    return data, labels

right_data_train, right_labels_train = preprocessing_training(right_data_train, right_labels_train)
right_data_val, right_labels_val = preprocessing_training(right_data_val, right_labels_val)


callbacks_for_search = [
    EarlyStopping(monitor='val_accuracy', patience=2),
    ModelCheckpoint(filepath='D:\\[COCA]_ASL_recognition\Model\\LSTM_parameter\\ModelCheckpoint\\model.{epoch:02d}-{val_loss:.2f}.h5'),
    TensorBoard(log_dir='D:\\[COCA]_ASL_recognition\Model\\LSTM_parameter\\TensorBoard_search'),
]

callbacks_for_right_train = [
    EarlyStopping(monitor='val_accuracy', patience=5),
    # ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
    TensorBoard(log_dir='D:\\[COCA]_ASL_recognition\Model\\LSTM_parameter\\TensorBoard_right_train'),
]

callbacks_for_left_train = [
    EarlyStopping(monitor='val_accuracy', patience=5),
    # ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
    TensorBoard(log_dir='D:\\[COCA]_ASL_recognition\Model\\LSTM_parameter\\TensorBoard_left_train'),
]
# !_________________________________________________________________

def build_model(hp):
    model = Sequential()

    batch_size = hp.Choice('batch_size', values=[128, 256])
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")

    model.add(
        LSTM(
            # Define the hyperparameter.
            units=hp.Int("LSTM1", min_value=128, max_value=256, step=32),
            return_sequences=True,
            activation="relu",
            input_shape=(1, 63)
        )
    )
    model.add(
        LSTM(
            units=hp.Int("LSTM2", min_value=64, max_value=128, step=32),
            return_sequences=True,
            activation="relu",
        )
    )
    model.add(
        LSTM(
            units=hp.Int("LSTM3", min_value=32, max_value=64, step=32),
            return_sequences=False,
            activation="relu",               
        )
    )
    model.add(
        Dense(
            # Define the hyperparameter.
            units=hp.Int("Dense1", min_value=64, max_value=128, step=32),
            activation="relu",
        )  
    )  
    if hp.Boolean("dropout"):
        model.add(Dropout(rate=0.25))

    model.add(
        Dense(
            # Define the hyperparameter.
            units=hp.Int("Dense2", min_value=64, max_value=128, step=32),
            activation="relu",
        )  
    ) 
    model.add(Dense(39, activation="softmax"))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate), 
        loss="categorical_crossentropy", 
        metrics=["accuracy"],
    )
    return model

tuner = keras_tuner.RandomSearch(
    hypermodel=build_model,
    objective="val_accuracy",
    max_trials=20,
    executions_per_trial=1,
    overwrite=True,
    directory="D:\\[COCA]_ASL_recognition\Model\\LSTM_parameter",
    project_name="[COCA]_ASL_recognition",
)

tuner.search_space_summary()

tuner.search(right_data_train, right_labels_train, validation_data=(right_data_val, right_labels_val), epochs=10, callbacks=callbacks_for_search)
tuner.results_summary()

right_hand_data, right_hand_labels = preprocessing_training(right_hand_data, right_hand_labels)
left_hand_data, left_hand_labels = preprocessing_training(left_hand_data, left_hand_labels)

best_hps = tuner.get_best_hyperparameters(3)
print(best_hps)
# Build the model with the best hp.
right_hand_model = build_model(best_hps[0])
right_hand_model.summary()
right_hand_model.fit(x=right_hand_data, y=right_hand_labels, epochs=50, callbacks=callbacks_for_right_train)

for layer in right_hand_model.layers:
    print(layer.name)
    print(layer.get_config())

left_hand_model = build_model(best_hps[0])
left_hand_model.summary()
left_hand_model.fit(x=left_hand_data, y=left_hand_labels, epochs=50, callbacks=callbacks_for_left_train)

for layer in left_hand_model.layers:
    print(layer.name)
    print(layer.get_config())

right_hand_model.save('D:\\[COCA]_ASL_recognition\Model\\right_hand_model.h5')
left_hand_model.save('D:\\[COCA]_ASL_recognition\Model\\left_hand_model.h5')

# # res = best_model.predict(x_test)
# res = right_hand_model.predict(x_test)
# res_true = np.argmax(y_test, axis=1)    

# # Đoán nhãn từ dự đoán của mô hình
# predicted_labels = np.argmax(res, axis=1)  # Lấy index của nhãn có xác suất cao nhất

# # So sánh với nhãn thực tế
# true_labels = np.argmax(y_test, axis=1)  # Giả sử y_test đã được one-hot encoding

# # Tính accuracy
# accuracy = np.mean(predicted_labels == true_labels)
# print(f'Accuracy: {accuracy * 100:.2f}%')


