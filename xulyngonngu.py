import pickle
import json
import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model
import spacy
# test 2 
# Load spaCy model




nlp = spacy.load("en_core_web_sm")
model = load_model("D:\\Project AI\\new_model_test\\new_model_test.h5")

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Đường dẫn đến file JSON chứa dữ liệu
file_path = "D:\\Project AI\\new_model_test\\Index_to_letters.json"

# Đọc dữ liệu từ file JSON
with open(file_path, 'r') as json_file:
    data = json.load(json_file)

# Tạo một dictionary đảo ngược từ file JSON
reversed_data = {v: k for k, v in data.items()}

# Tạo list mới với thứ tự đảo ngược
labels_dict =  [reversed_data[i] for i in range(len(reversed_data))]
# Constants for stability check
STABLE_FRAMES_THRESHOLD = 9
current_stable_frames = 0
stable_character = None
# Initialize a variable to store the accumulated word
accumulated_word = ""
# Khởi tạo biến để lưu ký tự trước đó
previous_character = None
while True:

    data_aux = []
    x_ = []
    y_ = []
    z_ = []
   
    ret, frame = cap.read()

    if ret:
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Các xử lý khác ở đây...
    else:
        continue  # Hoặc xử lý khác nếu frame không hợp lệ

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                z = hand_landmarks.landmark[i].z

                x_.append(x)
                y_.append(y)
                z_.append(z)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                z = z_[i]

                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))
                data_aux.append(z)


        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        input_data = np.array(data_aux)
       
        time_steps = 1  # Số bước thời gian trong dữ liệu, có thể là 1 tùy vào cách bạn xử lý dữ liệu
        reshaped_data = input_data.reshape(1, 1, len(input_data))  

        prediction = model.predict(reshaped_data)
        predicted_index = np.argmax(prediction)

        # Sử dụng index để truy xuất nhãn từ labels_dict
        predicted_character = labels_dict[predicted_index]
       
        if predicted_character == "space":
        # Kiểm tra xem ký tự trước đó có phải là khoảng trắng không
         if previous_character != "space":
            accumulated_word += " "
         elif predicted_character == "del":
            accumulated_word = accumulated_word[:-1]
        else:
            if predicted_character == stable_character:
                current_stable_frames += 1
                if current_stable_frames == STABLE_FRAMES_THRESHOLD:
                    accumulated_word += predicted_character
                    stable_window = np.ones((700, 700, 3), dtype=np.uint8) * 255
                    cv2.putText(stable_window, accumulated_word, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.imshow('Stable Character', stable_window)
                    cv2.waitKey(1)
                    print(f'Stable Character: {predicted_character}')
            else:
                stable_character = predicted_character
                current_stable_frames = 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)
        # Hiển thị kết quả dự đoán ra màn hình
        print(f'Predicted Character: {predicted_character}')
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Nhấn phím 'q' để thoát
        break

cap.release()
cv2.destroyAllWindows()
