import csv
import random
from PIL import Image, ImageOps
from PNRU_Extraction import noise_extract as prnu
from PNRU_Extraction import check_orientation
from Cross_Correlation_Functions import pce, crosscorr_2d
import numpy as np
import openpyxl as pyxl


labels = {} # Path : Device
paths = []
devices = {}
full_dataset = [['Image Path', 'Actual Class', 'Predicted Class']]

def create_devices_count(device_names):
    devices_count = {}
    for device in device_names:
        devices_count[device]=0
    return devices_count

def create_devices(device_names):
    count = 0
    for device in device_names:
        devices[device] = count
        devices[count] = device
        count += 1

def calculate_dataset_size(program_dataset):
    sizes = []
    for i in program_dataset:
        sizes.append(len(program_dataset[i]))
    smallest_count = min(sizes)
    return smallest_count

def load_equal_data(dataset_size, devices_count, program_dataset):
    this_devices_count = devices_count.copy()
    for device in program_dataset:
        this_paths = program_dataset[device]
        random.shuffle(this_paths)
        for path in this_paths:
            if this_devices_count[device] < dataset_size:
                labels[path] = device
                paths.append(path)
                this_devices_count[device] +=1
        
def create_equal_sized_folds(fold_size, device_data_quant, number_of_devices, devices_count):
    folded_data = []
    this_paths = paths.copy()
    random.shuffle(this_paths)
    fold_quant = 1 # I have opted to use the maximum ammount of data and only have 1 fold as this will lead to maximum accuracy. 

    print(f"{fold_quant} Folds Will Be Created, Each With {fold_size*number_of_devices} Entries.")
    print("-----------------------------------------------------------------------------------------------------------")

    if fold_quant > 0:
        for i in range(fold_quant):
            counter = 0
            this_devices_count = devices_count.copy()
            current_fold = []
            for image in this_paths:
                if (sum(this_devices_count.values())) < fold_size*number_of_devices:
                    if this_devices_count[labels[this_paths[counter]]] < fold_size:
                        current_fold.append(this_paths[counter])
                        this_devices_count[labels[this_paths[counter]]] += 1
                        counter += 1
                    else:
                        counter+=1
                else:
                    break
            for i in current_fold[:]:
                if i in this_paths:
                    this_paths.remove(i)
            folded_data.append(current_fold)
    return folded_data

def get_fold_stats(folds, device_count):
    for fold in folds:
        count = []
        for i in range(device_count):
            count.append(0)
        for image in fold:
            device = labels[image]
            count[devices[device]] += 1
    return count

def choose_suspect_image(folds):
    new_fold = []
    for fold in folds:
        current_fold = []
        current_fold.append(fold[0])
        del fold[0]
        current_fold.append(fold)
        new_fold.append(current_fold)
    return new_fold

def open_control(paths):
    control_images = []

    for i in paths:
        hold = Image.open(i)
        hold = ImageOps.exif_transpose(hold)
        hold = hold.convert('L')
        control_images.append(check_orientation(hold))
    return control_images

def open_suspect(path):
    hold = Image.open(path)
    hold = ImageOps.exif_transpose(hold)
    hold = hold.convert('L')
    suspect_image = check_orientation(hold)
    suspect_image = np.asarray(suspect_image)
    return suspect_image[:1000, :800]

def average(list): 
    if len(list) > 0:
        return sum(list) / len(list)
    else:
        return 0

def configure_environment(device_names, program_dataset):
    devices_count = create_devices_count(device_names)
    create_devices(device_names)
    smallest_dataset_size = calculate_dataset_size(program_dataset)
    return devices_count, smallest_dataset_size

def calculate(device_names, number_of_devices, program_dataset, suspect_image_paths):
    devices_count , smallest_dataset_size = configure_environment(device_names, program_dataset)
    load_equal_data(smallest_dataset_size, devices_count, program_dataset)
    folds = create_equal_sized_folds(smallest_dataset_size, smallest_dataset_size, number_of_devices, devices_count)
    get_fold_stats(folds, number_of_devices)
    #print(folds)

    # correct = 0
    # incorrect = 0

    for fold in folds:
        to_write = [] # path (of suspect image), actual class, predicted class, class scores
        control_fingerprints = []
        suspect_fingerprints = []
        scores = []
        for i in range(number_of_devices):
            scores.append([])
        avg_scores = []
        suspect_image_path = suspect_image_paths[0]
        control_images_paths = fold
        suspect_image = open_suspect(suspect_image_path[0])
        control_images = open_control(control_images_paths)

        suspect_prnu = prnu(suspect_image)

        prog = len(control_images)
        counter = 1
        for image in control_images:
            image = np.asarray(image)
            image = image[:1000, :800]
            control_fingerprints.append(prnu(image))
            #print(f"Progress: {round((counter/prog) * 100, 2)}%")
            counter+=1

        suspect_fingerprint = suspect_prnu
        count = 0
        tracker = 1
        for fingerprint in control_fingerprints:
            cross_cor = crosscorr_2d(suspect_fingerprint, fingerprint)
            pce_val = pce(cross_cor)
            pos = devices[labels[control_images_paths[count]]]
            scores[pos].append(pce_val['cc'])
            count += 1
            #print(f" Cross Correlation Progress: {round((tracker/prog) * 100, 2)}%")
            tracker+=1
        
        for i in range(number_of_devices):
            avg_scores.append(average(scores[i]))
        
        index_max = avg_scores.index(max(avg_scores))
        predicted_identity = (devices[index_max])
        print(f"Suspect Device : Unknown \nProgram Predicted Match: {predicted_identity}")
        print(avg_scores)
        # print(f"Scores: \nD3500: {average(scores[0])}   Drone: {average(scores[1])}   iPhone 8: {average(scores[2])}   iPhone 13: {average(scores[3])}   Samsung Galaxy S7 A: {average(scores[4])}   Samsung Galaxy S7 B: {average(scores[5])}")
        # print("--------------------------------------------------------------------------------------------------------------------------------------------------------------")
        # if suspect_name == predicted_identity:
        #     correct += 1
        #     verdict = "Correct"
        # else:
        #     incorrect +=1
        #     verdict = "Incorrect"
        # to_write.append([suspect_image_path, labels[suspect_image_path], predicted_identity, average(scores[0]), average(scores[1]), average(scores[2]), average(scores[3]), average(scores[4]), average(scores[5]), verdict])
        # full_dataset.append(to_write[0])
        # plt.xlabel("Device")
        # plt.ylabel("Match Likelihood")
        # plt.bar(device_names, avg_scores)
        # plt.grid()
        # plt.show()
    return avg_scores


def main(device_names, number_of_devices, program_dataset, test_runs, suspect_image_paths):
    run_average = np.zeros(number_of_devices)
    for i in range(test_runs):
        avg_scores = calculate(device_names, number_of_devices, program_dataset, suspect_image_paths)
        np_avg_scores = np.array(avg_scores)
        run_average = run_average + np_avg_scores
    final_scores = run_average/test_runs
    final_scores = final_scores.tolist()
    return final_scores