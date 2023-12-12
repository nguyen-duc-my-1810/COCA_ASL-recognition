import os
import shutil
import cv2
import mediapipe as mp

def extract_hand_landmarks(image_path):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    hands.close()
    return results.multi_hand_landmarks

def move_images_with_landmarks(input_directory, output_directory, limit_per_folder=5000):
    for root, dirs, files in os.walk(input_directory):
        for dir_name in dirs:
            source_directory = os.path.join(input_directory, dir_name)
            print(source_directory)
            destination_directory = os.path.join(output_directory, dir_name)
            print(destination_directory)
            if not os.path.exists(destination_directory):
                os.makedirs(destination_directory)

            image_files = [file for file in os.listdir(source_directory)]
            count = 0  # Biến đếm số lượng ảnh đã chuyển
            for image_file in image_files:
                if count >= limit_per_folder:
                    break  # Nếu đã chuyển đủ 100 ảnh, thoát vòng lặp
                source_image_path = os.path.join(source_directory, image_file)
                landmarks = extract_hand_landmarks(source_image_path)

                if landmarks:
                    destination_image_path = os.path.join(destination_directory, image_file)
                    shutil.copyfile(source_image_path, destination_image_path)
                    count += 1  # Tăng biến đếm khi chuyển một ảnh
            print(count)

# Đường dẫn thư mục "Test" chứa các thư mục con A, B, C, D...
input_test_directory = "D:\\New_data_ASL\\asl_alphabet_train"

# Đường dẫn thư mục "Train" để chứa ảnh có thể nhận diện landmark
output_train_directory = "D:\\Test_ASL_Model_YTB\\sign-language-detector-python-master\\New_data"

move_images_with_landmarks(input_test_directory, output_train_directory)

#! Lúc sau chỉnh lại thành 3 đoạn như này