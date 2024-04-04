# Required imports for interface and image processing
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import random
from PIL import Image, ImageTk, ImageOps
from PNRU_Extraction import main, check_orientation
from Cross_Correlation_Functions import pce, crosscorr_2d
from Device_Identification import main as ident_main
from scipy.signal import correlate2d
import matplotlib.pyplot as plt
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,  
NavigationToolbar2Tk) 

# Setting the size the image thumbnails will have
size = 180, 180

# Global store of images
target_image = [] # This will store the Pillow 'Image' type of the target image
image_dataset = [] # This will store all of the images in the dataset as Pillow 'Image' Types
prnu_fingerprints_display = [] # [suspect image (Pil Img), Control image (Pil Img)] This will store the two image fingerprints for diplay purposes, the first will be the suspect image print and the second the control images
prnu_fingerprints = [] # [suspect image (np.array), Control image (np.array)] This will store the raw fingerprints for processing purposes.

# This allows the user to select images from their file system to build the image set that will be analysed
def open_set_sample():
    image_dataset.clear() # Empties the storage array each time new images are selected
    reset_sample()
    reset_content()
    # returns a tuple of the file paths
    file_path = filedialog.askopenfilenames(title="Select Control Images", filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.ico")])
    if file_path:
        display_image_sample(file_path)
        img_dataset = list(file_path) # Converting the tuple to a list for iteration
        for image in img_dataset:
            hold = Image.open(image)
            hold = ImageOps.exif_transpose(hold)
            image_dataset.append(check_orientation(hold)) # Adding the Pillow 'Image' types to a list
    else:
         messagebox.showerror('File Error', 'Error: Please Select Min. 1 Valid Image')

# This allows the user to select a target image that wil be used for comparison
def open_target_image():
    target_image.clear() # Empties the storage array each time new images are selected
    reset_target()
    reset_content()
    # returns a tuple of the file paths
    file_path = filedialog.askopenfilenames(title="Select Suspect Images", filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.ico")])
    if file_path:
        display_target_image(file_path)
        img_dataset = list(file_path) # Converting the tuple to a list for iteration
        for image in img_dataset:
            hold = Image.open(image)
            hold = ImageOps.exif_transpose(hold)
            # print(f"hold2 size: {hold2.size}")
            # print("-----------------------")
            target_image.append(check_orientation(hold)) # Adding the Pillow 'Image' types to a list  
    else:
        messagebox.showerror('Warning', 'No Image Selected (Optional)')     

# All this does is choose a random sample from the image data set
def select_random_display(file_path):
    set_length = len(file_path)
    if set_length > 4:
        randints = random.sample(range(0,(len(file_path)-1)),4) # This simply selects 4 images from the dataset
    else:
        randints = list(range(0,set_length)) # If there are less than 4 images in the dataset this will display all of them
    return_list = []
    for value in randints:
        return_list.append(file_path[value])
    return return_list

# This function merely displays the sample of the dataset
def display_image_sample(file_path):
    row = 2
    col = 1
    images = select_random_display(file_path)

    reset_sample() # This will clear the displayed thumbnails

    for item in images:
        image =Image.open(item)
        image = ImageOps.exif_transpose(image)
        image = image.resize((size), Image.Resampling.LANCZOS)
        image = ImageTk.PhotoImage(image)
        label = tk.Label(random_sample_frame, image=image)
        label.photo = image 
        label.grid(row=row, column=col, padx=0, pady=0)
        sample_labels.append(label)
        col += 1

# This function displays the target or control image
def display_target_image(file_path):
    row = 2
    col = 1
    images = select_random_display(file_path)

    reset_target() # This will clear the displayed thumbnails

    for item in images:
        print(item)
        image =Image.open(item)
        image = ImageOps.exif_transpose(image)
        image = image.resize((size), Image.Resampling.LANCZOS)
        image = ImageTk.PhotoImage(image)
        label = tk.Label(target_image_frame, image=image)
        label.photo = image 
        label.grid(row=row, column=col, padx=0, pady=0)
        target_labels.append(label)
        col += 1

# This function displays the target image after it has been manipulated
def display_computed_image(images, display):
    row = 0
    col = 0

    reset_content()
    if len(images) > 1:
        for item in images:
            image = item.resize((display), Image.Resampling.LANCZOS)
            image = ImageTk.PhotoImage(image)
            label = tk.Label(content_frame, image=image)
            label.photo = image 
            label.grid(row=row, column=col, padx=0, pady=0)
            content_labels.append(label)
            if col == 9:
                col = 0
                row +=1
            else:
                col += 1
    else:
        image = images[0].resize((display), Image.Resampling.LANCZOS)
        image = ImageTk.PhotoImage(image)
        label = tk.Label(content_frame, image=image)
        label.photo = image 
        label.grid(row=row, column=col, padx=0, pady=0)
        content_labels.append(label)

# This function is used to clear the thumbnails of the random sample
def reset_sample():
    for label in sample_labels:
        label.destroy()

# This function is used to clear the contents of the main application frame
def reset_content():
    for label in content_labels:
        label.destroy()

# This function is used to clear the thumbnails of the target images sample
def reset_target():
    for label in target_labels:
        label.destroy()

# This is a temporary function used to test the systems ability to manipulate and display images
def compute_test():
    greyscale_images = []
    if (len(target_image)>0): # Checking that there is an image selected for computing
        for img in target_image:
            greyscale_images.append(img.convert('L'))
        display_computed_image(greyscale_images, (200,200))
    else:
        messagebox.showerror('Computation Error', 'Error: Please Select Analysis Image First') # pop up error box

# This is a support function used to convert the images to greyscale
def compute_greyscale(set):
    greyscale_images = []
    for img in set:
        #print(img.size)
        greyscale_images.append(img.convert('L'))
    #print(f" grey scale size: {greyscale_images[0].size}")
    return greyscale_images

# This fuction meerly calls the denoising function in another module
def denoise_target():
    if (len(target_image)>0): # Checking that there is an image selected for computing
        image_set = compute_greyscale(target_image)
        denoised = main(image_set)
        #display_computed_image(denoised[0], (600,600))
        prnu_fingerprints_display.append(denoised[0][0])
        prnu_fingerprints.append(denoised[1][0])
    else:
        messagebox.showerror('Computation Error', 'Error: Please Select Analysis Image First') # pop up error box
    
    if (len(image_dataset)>0): # Checking that there is an image selected for computing
        image_set = compute_greyscale(image_dataset)
        denoised = main(image_set)
        #display_computed_image(denoised[0], (600,600))
        prnu_fingerprints_display.append(denoised[0][0])
        prnu_fingerprints.append(denoised[1][0])
    else:
        messagebox.showerror('Computation Error', 'Error: Please Select Control Image First') # pop up error box


    display_computed_image(prnu_fingerprints_display, (600,600))
    #cross_cor = aligned_cc(prnu_fingerprints[0], prnu_fingerprints[1])
    fp_1 = prnu_fingerprints[0]
    fp_2 = prnu_fingerprints[1]
    cross_cor = crosscorr_2d(fp_1, fp_2)
    pce_val = pce(cross_cor)
    print(f"Peak: {pce_val['peak']} PCE: {pce_val['pce']} CC: {pce_val['cc']}")

    prnu_fingerprints.clear()
    prnu_fingerprints_display.clear()

# This function is used for the main body of this project
def identify_device():
    # print(devices_count)
    device_names = []
    device_images=[]
    suspect_images = []
    program_data = {}

    while True:
        suspect_name = simpledialog.askstring(title= f"Suspect Device", prompt=f"Suspect Device Name TESTING:")
        suspect_path = filedialog.askopenfilenames(title=f"Select Suspect Image(s)", filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.ico")])
        if suspect_path:
            suspect_images.append(list(suspect_path))
            display_target_image(suspect_images[0])
            break
        else:
            messagebox.showerror('File Error', 'Error: Please Select Min. 1 Valid Image')

    devices_count = simpledialog.askinteger(title="Number Of Devices", prompt="Please enter the number of control devices:")

    for i in range(devices_count):
        while True:
            device_name = simpledialog.askstring(title= f"Device #{i+1} Name", prompt=f"Enter Device {i+1} Name:")
            if device_name in device_names:
                messagebox.showerror('Upload Error', f'The name {device_name} has already been used!')
            elif len(device_name) < 1:
                messagebox.showerror('Upload Error', 'The name cannot be blank!')
            else:
                device_names.append(device_name)
                file_path = filedialog.askopenfilenames(title=f"Select {device_name} Images", filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.ico")])
                if file_path:
                    img_dataset = list(file_path)
                    for img in img_dataset:
                        device_images.append(img)
                    program_data[device_name]=img_dataset
                else:
                    messagebox.showerror('File Error', 'Error: Please Select Min. 1 Valid Image')
                break
        display_image_sample(device_images)
    while True:
        test_runs = simpledialog.askinteger(title="Number Of Test Rounds", prompt="Please enter the number of rounds of testing desired \n(Minimum 50 Recommended):")
        if test_runs == 0:
            messagebox.showerror('Quantity Error!', 'The Quantity Must Be Greater Than 0')
        else:
            break
    # print(device_names)
    # print(device_images)
    # print(program_data)
    print("----------------------------------- Device Identification Below ---------------------------------------")
    scores = ident_main(device_names, devices_count, program_data, test_runs, suspect_images, suspect_name)
    # plt.bar(device_names, scores)
    # plt.grid()
    # plt.show()
    plot(device_names, scores)

def plot(device_names, scores):
    fig = Figure(figsize = (7, 4)) 
    plot1 = fig.add_subplot(111)
    plot1.set_title('Suspect Device Likelihood')
    plot1.grid()
    plot1.set_xlabel("Device")
    plot1.set_ylabel("Match Likelihood")
    plot1.bar(device_names, scores)
    canvas = FigureCanvasTkAgg(fig, master = content_frame)   
    canvas.draw()
    canvas.get_tk_widget().pack() 

# This function create all the labels used within the interface
def create_labels():
    random_display_label = tk.Label(random_sample_frame, text="Random Sample Of Control Set", font=("Helvetica", 12))
    random_display_label.grid(row=0, column=1, sticky="nsew", columnspan=4) 

    target_image_label = tk.Label(target_image_frame, text="Random Sample Of Suspect Set", font=("Helvetica", 12))
    target_image_label.grid(row=0, column=1, sticky="nsew", columnspan=4) 

# This function creates all the buttons for the interface
def create_buttons():
    open_analysis_set_btn = tk.Button(main_button_frame, text="Select Control Set", command=open_set_sample)
    open_analysis_set_btn.grid(row=1, column=0)  

    open_target_image_btn = tk.Button(main_button_frame, text="Select Suspect Image", command=open_target_image)
    open_target_image_btn.grid(row=1, column=1)  

    reset_btn = tk.Button(main_button_frame, text="Reset Selection", command=reset_all)
    reset_btn.grid(row=1, column=2)

    compute_btn = tk.Button(main_button_frame, text="Compute (Greyscale)", command=compute_test)
    compute_btn.grid(row=1, column=3)

    denoise_btn = tk.Button(main_button_frame, text="Denoise (WIP)", command=denoise_target)
    denoise_btn.grid(row=1, column=4)

    identify_btn = tk.Button(main_button_frame, text="Identify Device", command=identify_device)
    identify_btn.grid(row=1, column=5)

# This function resets all of the thumbnails and storage lists for data selected in the program
def reset_all():
    for label in all_labels:
        label.destroy() # This will just clear all of the images selected by the user
    reset_sample()
    reset_target()
    reset_content()
    target_image.clear()
    image_dataset.clear()

# Does thing for Tkinter
all_labels = []
sample_labels = []
target_labels = []
content_labels = []
root = tk.Tk()

# Set window title
root.title("Image Analysis Prototype")

# Set app to full screen on startup
width= root.winfo_screenwidth() 
height= root.winfo_screenheight()
root.geometry("%dx%d" % (width, height))

# Below are all of the frames used for this application
# This is the frame for all of the buttons in the top of the application
main_button_frame = tk.Frame(root, relief="flat", borderwidth=1)
main_button_frame.grid(row=0, column=0, sticky = "nw", columnspan=2)

# This frame will be used for the main section of the screen
content_frame = tk.Frame(root, background="lightgray", relief="flat", borderwidth=1, highlightbackground="black", highlightthickness=2)
content_frame.grid(row=1, column=0, sticky = "nsew", columnspan=2)

# This frame is for the target image display in the lower left
target_image_frame = tk.Frame(root, background="lightgray", padx=10, pady=10, relief="flat", borderwidth=1)
target_image_frame.grid(row=2, column=0, sticky="sw")
target_image_frame.columnconfigure(0, weight=1)
target_image_frame.rowconfigure(0, weight=1)

# This frame is for the random sample of images in the lower right
random_sample_frame = tk.Frame(root, background="lightgray", padx=10, pady=10, relief="flat", borderwidth=1)
random_sample_frame.grid(row=2, column=1, sticky="se")
random_sample_frame.columnconfigure(0, weight=1)
random_sample_frame.rowconfigure(0, weight=1)

# Configuring the grid 
root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(1, weight=1)

# General root configuration
root.configure(bg="lightgray")

# Generating Labels and buttons
create_labels()
create_buttons()

root.mainloop()       