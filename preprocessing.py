import pickle
import numpy as np
from collections import Counter
import random

import Augmentation

data_right_hand = pickle.load(open("D:\\[COCA]_ASL_recognition\Model\\new_data.pickle", 'rb'))
data_right_hand = data_right_hand

data_left_hand = pickle.load(open("D:\\[COCA]_ASL_recognition\Model\\flip_data.pickle", 'rb'))
data_left_hand = data_left_hand

label_counts = Counter(data_right_hand["labels"])
sorted_labels = sorted(label_counts.items(), key=lambda x: x[0])
print(sorted_labels)

label_counts = Counter(data_left_hand["labels"])
sorted_labels = sorted(label_counts.items(), key=lambda x: x[0])
print(sorted_labels)

def augmentation_implementation(data):
    # ! Resampling dữ liệu lên
    for label in range(28):
        str_label = str(label)
        
        filtered_indices = [idx for idx, lbl in enumerate(data['labels']) if lbl == str_label]
        filtered_data = [data['data'][idx] for idx in filtered_indices]

        # Nếu độ dài danh sách nhãn hiện tại nhỏ hơn 5000, thực hiện resampling
        if len(filtered_data) > 4500 and len(filtered_data) < 5000:
            # Resampling lên 5000 phần tử
            resampled_indices = np.random.choice(filtered_indices, size=5000 - len(filtered_data), replace=True)
            resampled_data = [data['data'][idx] for idx in resampled_indices]

            # Gán lại các phần tử đã resampled vào vị trí ban đầu trong dữ liệu
            data['data'].extend(resampled_data)
            data['labels'].extend([str_label] * (5000 - len(filtered_data)))
        
        if len(filtered_data) > 3500 and len(filtered_data) < 4000:  
            resampled_indices = np.random.choice(filtered_indices, size=4000 - len(filtered_data), replace=True)
            resampled_data = [data['data'][idx] for idx in resampled_indices]

            data['data'].extend(resampled_data)
            data['labels'].extend([str_label] * (4000 - len(filtered_data)))

        if len(filtered_data) < 3500:
            resampled_indices = np.random.choice(filtered_indices, size=3500 - len(filtered_data), replace=True)
            resampled_data = [data['data'][idx] for idx in resampled_indices]

            data['data'].extend(resampled_data)
            data['labels'].extend([str_label] * (3500 - len(filtered_data)))
    

    for label in range(28, 39):
        str_label = str(label)
        
        filtered_indices = [idx for idx, lbl in enumerate(data['labels']) if lbl == str_label]
        filtered_data = [data['data'][idx] for idx in filtered_indices]

        if len(filtered_data) > 1500 and len(filtered_data) < 1700:   
            resampled_indices = np.random.choice(filtered_indices, size=1700 - len(filtered_data), replace=True)
            resampled_data = [data['data'][idx] for idx in resampled_indices]

            data['data'].extend(resampled_data)
            data['labels'].extend([str_label] * (1700 - len(filtered_data)))

        if len(filtered_data) > 1000 and len(filtered_data) < 1500:      
            resampled_indices = np.random.choice(filtered_indices, size=1500 - len(filtered_data), replace=True)
            resampled_data = [data['data'][idx] for idx in resampled_indices]

            data['data'].extend(resampled_data)
            data['labels'].extend([str_label] * (1500 - len(filtered_data)))

        if str_label == '28':
            resampled_indices = np.random.choice(filtered_indices, size=300 - len(filtered_data), replace=True)
            resampled_data = [data['data'][idx] for idx in resampled_indices]

            data['data'].extend(resampled_data)
            data['labels'].extend([str_label] * (300 - len(filtered_data)))

        if str_label == '33':
            resampled_indices = np.random.choice(filtered_indices, size=1800 - len(filtered_data), replace=True)
            resampled_data = [data['data'][idx] for idx in resampled_indices]

            data['data'].extend(resampled_data)
            data['labels'].extend([str_label] * (1800 - len(filtered_data)))

        if str_label == '38':
            resampled_indices = np.random.choice(filtered_indices, size=1000 - len(filtered_data), replace=True)
            resampled_data = [data['data'][idx] for idx in resampled_indices]

            data['data'].extend(resampled_data)
            data['labels'].extend([str_label] * (1000 - len(filtered_data)))

    label_counts = Counter(data["labels"])
    print(label_counts)

    # ! Phép đổi góc nhìn, xoay 3D
    rotate_data = []
    rotate_labels = []

    for label in range(39):
        str_label = str(label)
        
        filtered_indices = [idx for idx, lbl in enumerate(data['labels']) if lbl == str_label]
        filtered_data = [data['data'][idx] for idx in filtered_indices]

        if len(filtered_data) == 5000:
            data_rotate_number = 500
        elif len(filtered_data) == 4000:
            data_rotate_number = 400
        elif len(filtered_data) == 3500:
            data_rotate_number = 450
        elif len(filtered_data) == 1700 or len(filtered_data) == 1500:
            data_rotate_number = 150
        else:
            data_rotate_number = 50

        selected_samples = random.sample(filtered_data, min(data_rotate_number, len(filtered_data)))
        
        data_rotate_final = []

        for i in range(len(selected_samples)):
            divided_samples = selected_samples[i]
            data_rotate = Augmentation.change_camera_perspective(divided_samples, 10, 'x')
            data_rotate_final.append(data_rotate)

        selected_samples = random.sample(filtered_data, min(data_rotate_number, len(filtered_data)))
        
        for i in range(len(selected_samples)):
            divided_samples = selected_samples[i]
            data_rotate = Augmentation.change_camera_perspective(divided_samples, -10, 'x')
            data_rotate_final.append(data_rotate)

        selected_samples = random.sample(filtered_data, min(data_rotate_number, len(filtered_data)))

        for i in range(len(selected_samples)):
            divided_samples = selected_samples[i]
            data_rotate = Augmentation.change_camera_perspective(divided_samples, 10, 'y')
            data_rotate_final.append(data_rotate)

        selected_samples = random.sample(filtered_data, min(data_rotate_number, len(filtered_data)))

        for i in range(len(selected_samples)):
            divided_samples = selected_samples[i]
            data_rotate = Augmentation.change_camera_perspective(divided_samples, -10, 'y')
            data_rotate_final.append(data_rotate)

        selected_samples = random.sample(filtered_data, min(data_rotate_number, len(filtered_data)))
    
        for i in range(len(selected_samples)):
            divided_samples = selected_samples[i]
            data_rotate = Augmentation.change_camera_perspective(divided_samples, 10, 'z')
            data_rotate_final.append(data_rotate)

        selected_samples = random.sample(filtered_data, min(data_rotate_number, len(filtered_data)))

        for i in range(len(selected_samples)):
            divided_samples = selected_samples[i]
            data_rotate = Augmentation.change_camera_perspective(divided_samples, -10, 'z')
            data_rotate_final.append(data_rotate)

        rotate_data.extend(data_rotate_final)
        rotate_labels.extend([str_label] * len(data_rotate_final))

    # ! Chuyển
    change_data = []
    change_labels = []
    for label in range(39):
        str_label = str(label)
        
        filtered_indices = [idx for idx, lbl in enumerate(data['labels']) if lbl == str_label]
        filtered_data = [data['data'][idx] for idx in filtered_indices]
        # print(len(filtered_data))

        if len(filtered_data) == 5000:
            data_change_ratio_number = 500
        elif len(filtered_data) == 4000:
            data_change_ratio_number = 400
        elif len(filtered_data) == 3500:
            data_change_ratio_number = 450
        elif len(filtered_data) == 1700 or len(filtered_data) == 1500:
            data_change_ratio_number = 150
        else:
            data_change_ratio_number = 50

        selected_change_samples = random.sample(filtered_data, min(data_change_ratio_number, len(filtered_data)))

        data_change_ratio_final = []

        for i in range(len(selected_change_samples)):
            divided_samples = selected_change_samples[i]

            ratio_x = random.uniform(0.95, 0.99)
            ratio_y = random.uniform(0.95, 0.99)

            data_change_ratio = Augmentation.change_hand_ratio(divided_samples, ratio_x, ratio_y)
            data_change_ratio_final.append(data_change_ratio)

        selected_change_samples = random.sample(filtered_data, min(data_change_ratio_number, len(filtered_data)))

        for i in range(len(selected_change_samples)):
            divided_samples = selected_change_samples[i]

            ratio_x_negative = random.uniform(-0.95, -0.99)
            ratio_y_negative = random.uniform(-0.95, -0.99)

            data_change_ratio = Augmentation.change_hand_ratio(divided_samples, ratio_x_negative, ratio_y_negative)
            data_change_ratio_final.append(data_change_ratio)   

        change_data.extend(data_change_ratio_final)
        change_labels.extend([str_label] * len(data_change_ratio_final))

    # !
    add_minor_data = []
    add_minor_labels = []

    for label in range(39):
        str_label = str(label)
        filtered_indices = [idx for idx, lbl in enumerate(data['labels']) if lbl == str_label]
        filtered_data = [data['data'][idx] for idx in filtered_indices]

        if len(filtered_data) == 5000:
            data_add_minor_number = 500
        elif len(filtered_data) == 4000:
            data_add_minor_number = 400
        elif len(filtered_data) == 3500:
            data_add_minor_number = 450
        elif len(filtered_data) == 1700 or len(filtered_data) == 1500:
            data_add_minor_number = 150
        else:
            data_add_minor_number = 50

        selected_add_minor_samples = random.sample(filtered_data, min(data_add_minor_number, len(filtered_data)))

        data_add_minor_final = []

        for i in range(len(selected_add_minor_samples)):
            divided_samples = selected_add_minor_samples[i]

            x_magnitude = 0.005
            y_magnitude = 0.005
            z_percentage = 0.02

            data_add_minor = Augmentation.add_minor_fluctuations(divided_samples, x_magnitude, y_magnitude, z_percentage)
            data_add_minor_final.append(data_add_minor)


        selected_add_minor_samples = random.sample(filtered_data, min(data_add_minor_number, len(filtered_data)))
        
        for i in range(len(selected_add_minor_samples)):
            divided_samples = selected_add_minor_samples[i]

            x_magnitude = 0.004
            y_magnitude = 0.004
            z_percentage = 0.01

            data_add_minor = Augmentation.add_minor_fluctuations(divided_samples, x_magnitude, y_magnitude, z_percentage)
            data_add_minor_final.append(data_add_minor)

        add_minor_data.extend(data_add_minor_final)
        add_minor_labels.extend([str_label] * len(data_add_minor_final))

    data['data'].extend(rotate_data)
    data['labels'].extend(rotate_labels)

    data['data'].extend(change_data)
    data['labels'].extend(change_labels)

    data['data'].extend(add_minor_data)
    data['labels'].extend(add_minor_labels)



augmentation_implementation(data_right_hand)
right_hand_path = "D:\\[COCA]_ASL_recognition\\Model\\right_hand_data.pickle"  
with open(right_hand_path, 'wb') as file:
    pickle.dump(data_right_hand, file)

augmentation_implementation(data_left_hand)
left_hand_path = "D:\\[COCA]_ASL_recognition\\Model\\left_hand_data.pickle"  
with open(left_hand_path, 'wb') as file:
    pickle.dump(data_left_hand, file)

print(len(data_right_hand['data']))
label_counts = Counter(data_right_hand["labels"])
sorted_labels = sorted(label_counts.items(), key=lambda x: x[0])
print(sorted_labels)

label_counts = Counter(data_left_hand["labels"])
sorted_labels = sorted(label_counts.items(), key=lambda x: x[0])
print(sorted_labels)




