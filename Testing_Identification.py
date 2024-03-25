import csv
import random
from PIL import Image, ImageOps
from PNRU_Extraction import noise_extract as prnu
from PNRU_Extraction import check_orientation
from Cross_Correlation_Functions import pce, crosscorr_2d
import numpy as np


labels = {} # Path : Device
paths = []
devices = {'D3500': 0, 'Drone': 1, 'iPhone 8' : 2, 'iPhone13' : 3, 'Samsung Galaxy S7 A' : 4, 'Samsung Galaxy S7 B' : 5, 0 : 'D3500', 1 : 'Drone', 2 : 'iPhone 8', 3 : 'iPhone 13', 4 : 'Samsung Galaxy S7 A', 5 : 'Samsung Galaxy S7 B' }

def load_data():
    with open(r"C:\Users\Elliot\Documents\Comp Sci Year 3\Final Year Project\Classified_Data.csv", mode="r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            labels[row[0]] = row[1]
            paths.append(row[0])

def create_fold(folds):
    folded_data = []
    random.shuffle(paths)
    fold_size = (len(labels))//folds # Ammount of entries in each fold
    remainder = (len(labels))%folds # Ammount of left of entries
    fold_quant = (len(labels))//fold_size
    counter = 0
    for x in range(fold_quant):
        current_fold = []
        for i in range(fold_size):
            current_fold.append(paths[counter])
            counter +=1
        folded_data.append(current_fold)
    if remainder > 0:
        current_fold = []
        for i in range(remainder):
            current_fold.append(paths[counter])
            counter +=1
        folded_data.append(current_fold)
    return folded_data

def get_fold_stats(folds):
     for fold in folds:
        count = [0,0,0,0,0,0]
        for image in fold:
            device = labels[image]
            count[devices[device]] += 1
        print(f"D3500: {count[0]} Drone: {count[1]}, iPhone 8: {count[2]}, iPhone 13: {count[3]}, Samsung Galaxy S7 A: {count[4]}, Samsung Galaxy S7 B: {count[5]}") 
        

def choose_suspect_image(folds):
    # Do things
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
    return suspect_image[:800, :1000]

def average(list): 
    return sum(list) / len(list)

def main():
    load_data()
    folds = create_fold(5)
    get_fold_stats(folds)
    folds = choose_suspect_image(folds)
     # folds[0] will contain ['suspect image path', [list of control image paths]]
    print("--------------------------------------------------------------------------------------------------")
    for fold in folds:
        control_fingerprints = []
        scores = [[],[],[],[],[],[]]
        avg_scores = []
        suspect_image_path = fold[0]
        control_images_paths = fold[1]
        suspect_image = open_suspect(suspect_image_path)
        control_images = open_control(control_images_paths)

        suspect_prnu = prnu(suspect_image)
        #print("Suspect Calculated")
        prog = len(control_images)
        counter = 1
        for image in control_images:
            image = np.asarray(image)
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
        
        for i in range(5):
            avg_scores.append(average(scores[i]))
        
        index_max = avg_scores.index(max(avg_scores))
        predicted_identity = (devices[index_max])
        print(f"Suspect Device : {labels[suspect_image_path]}  \nProgram Predicted Match: {predicted_identity}")
        print(f"Scores: \nD3500: {average(scores[0])}   Drone: {average(scores[1])}   iPhone 8: {average(scores[2])}   iPhone 13: {average(scores[3])}   Samsung Galaxy S7 A: {average(scores[4])}   Samsung Galaxy S7 B: {average(scores[5])}")
        print("--------------------------------------------------------------------------------------------------------------------------------------------------------------")

main()