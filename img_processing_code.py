import streamlit as st
import cv2
from nlp_model import chat
from PIL import Image
from io import BytesIO
import numpy as np
import re
import os
import torch

def download():
    file_name = "output.jpg"
            
    new_edited_img.save(file_name)

    with open(file_name, "rb") as file:
        btn = st.download_button(
                    label="Download image",
                    data=file,
                    file_name=file_name,
                    mime="image/png"
                )

    os.remove(file_name)

st.write("Hello !!! In this current version of the project, the available tasks are ....")
st.write("1. Image Classification (i.e detecting female, male, top, pant, dress, hat, shoes, suit) \n2. Image Editing (i.e (1) adjusting brightness, (2) resize, (3) blur, (4) rotate, (5) crop) \n")


uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    # Open the image file
    image_bytes = uploaded_file.read()
    image = Image.open(BytesIO(image_bytes))

    # Display the original image
    st.image(image, caption="Original Image")

tx = st.text_input("Prompt", )
results_list, num_from_prommpt = chat(tx)

if len(results_list) == 0:
    command_list = ["rotate", "blur", "resize", "brightness", "detect", "crop"]

    counter = 0
    for i in command_list:
        if i in tx:
            results_list.append(i)
            counter = counter + 1
            # print(results_list)
        
    if counter == 0:
      st.write("Please enter the correct function")
    
    else:
        for i, tasks in enumerate(results_list):

            if tasks == "rotate":
                def rotate_function(image, degree):
                    if degree == 180:
                        output = cv2.rotate(image, cv2.ROTATE_180)

                    elif degree == 90:
                        output = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

                    elif degree == 270:
                        output = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

                    else:
                        output = image

                    return output
                    
                img_cv = np.array(image)

                # Rotate the image
                if len(num_from_prommpt) == 1:
                    index = min(i, len(num_from_prommpt) - 1)
                    value = num_from_prommpt[index]
                    rotated_img_cv = rotate_function(img_cv, value)
                
                elif len(num_from_prommpt) == 0 or len(num_from_prommpt) > 1:
                    degree_value = st.selectbox('choose the degree to rotate',('90', '180', '270'))
                    rotated_img_cv = rotate_function(img_cv, int(degree_value))

                # Convert the rotated image back to PIL format
                new_edited_img = Image.fromarray(rotated_img_cv)
                st.image(new_edited_img, caption="New image")
                download()



            elif tasks == "blur":
                def stack_blur(image, blur_percentage):
                
                    # Check if blur percentage is 0, return the original image
                    if blur_percentage == 0:
                        return image

                    # Calculate the blur factor based on the image size and blur percentage
                    image_height, image_width, _ = image.shape
                    blur_factor = int((max(image_height, image_width) / 100.0) * blur_percentage)

                    # Apply the stack blur effect
                    blurred_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    blurred_image = cv2.blur(blurred_image, (blur_factor, blur_factor))
                    blurred_image = cv2.cvtColor(blurred_image, cv2.COLOR_RGB2BGR)

                    return blurred_image
                

                img_cv = np.array(image)

                if len(num_from_prommpt) == 1:
                    index = min(i, len(num_from_prommpt) - 1)
                    value = num_from_prommpt[index]
                    rotated_img_cv = stack_blur(img_cv, value)
                
                elif len(num_from_prommpt) == 0 or len(num_from_prommpt) > 1:
                    blur_percent = int(st.slider('adjust the blur percentage', 0, 100))
                    rotated_img_cv = stack_blur(img_cv, blur_percent)

                new_edited_img = Image.fromarray(rotated_img_cv)

                # Display the rotated image
                st.image(new_edited_img, caption="New image")
                download()




            elif tasks == "brightness":
                def modify_brightness(image, brightness_percentage):
                    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

                    h, s, v = cv2.split(hsv_image)

                    brightness_factor = float(brightness_percentage) / 100.0
                    v = np.clip(v * brightness_factor, 0, 255).astype(np.uint8)
                    hsv_image = cv2.merge([h, s, v])

                    modified_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
                    return modified_image

                
                img_cv = np.array(image)

                if len(num_from_prommpt) == 1:
                    index = min(i, len(num_from_prommpt) - 1)
                    value = num_from_prommpt[index]
                    rotated_img_cv = modify_brightness(img_cv, value)
                
                else:
                    brightness_value = int(st.slider('adjust the brightness percentage', 0, 200))
                    rotated_img_cv = modify_brightness(img_cv, brightness_value)

                new_edited_img = Image.fromarray(rotated_img_cv)

                # Display the rotated image
                st.image(new_edited_img, caption="New image")      
                download()



            elif tasks == "crop":         
                def image_crop(image):
                    height, width, channel= image.shape

                    #first_num, second_num = ratio.split(":")

                    a = width
                    b = height

                    new_height_a = st.slider('adjust the height_a', 0, int(height))
                    new_height_b = st.slider('adjust the height_b', int(height), 1)
                    new_width_a = st.slider('adjust the width_a', 0, int(width))
                    new_width_b = st.slider('adjust the width_b', int(width), 1)

                    edit_img = cv2.rectangle(image, (new_width_a,new_height_a) , (new_width_b, new_height_b), (255,255,255), 3)
                    st.image(edit_img)
                    #st.button('Ok')

                    st.write("This is the output image. You can adjust the slide before download")

                    if new_height_a >= new_height_b:
                        if new_width_a >= new_width_b:
                            ah = new_height_a
                            bh = new_height_b
                            aw = new_width_a
                            bw = new_width_b

                        else:
                            ah = new_height_a
                            bh = new_height_b
                            aw = new_width_b
                            bw = new_width_a


                    else:
                        if new_width_a >= new_width_b:
                            ah = new_height_b
                            bh = new_height_a
                            aw = new_width_a
                            bw = new_width_b


                        else:
                            ah = new_height_b
                            bh = new_height_a
                            aw = new_width_b
                            bw = new_width_a


                    crop_image = edit_img[bh: ah, bw: aw]


                    return crop_image

                img_cv = np.array(image)

                # if len(num_from_prommpt) == 0:
                #     rotated_img_cv = image_crop(img_cv)
                
                # else:
                #     st.write("Write correct prompt")

                    # Convert the rotated image back to PIL format
                new_edited_img = Image.fromarray(rotated_img_cv)

                    # Display the rotated image
                st.image(new_edited_img, caption="New image")
                download()


            elif tasks == "resize":
                def resize_function(image, percentage):
                    width = int(image.shape[1] * percentage/100)
                    height = int(image.shape[0] * percentage/100)
                    new_size = (width, height)
                    resized_img = cv2.resize(image, new_size, interpolation = cv2.INTER_AREA)
                    return resized_img
                
                img_cv = np.array(image)

                if len(num_from_prommpt) == 1:
                    index = min(i, len(num_from_prommpt) - 1)
                    value = num_from_prommpt[index]
                    rotated_img_cv = resize_function(img_cv, value)
            
                else:
                    resize_value = st.slider('adjust the percentage to resize', 1, 200)
                    rotated_img_cv = resize_function(img_cv, resize_value)
                    
 
                new_edited_img = Image.fromarray(rotated_img_cv)

                    # Display the rotated image
                st.image(new_edited_img, caption="New image")   
                download()

            elif tasks == "detect":

                def resize_image(image, width, height):
                    resized_image = image.resize((width, height))
                    return resized_image
                        
                # Load the YOLOv5 model
                model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

                # Define a function to perform object detection on the uploaded image
                def perform_object_detection(image):
                
                    results = model(image)

                    annotated_image = Image.fromarray(results.render()[0])

                    return annotated_image
                
                #img_cv = np.array(image)
                original_width, original_height = image.size

                resized_image = resize_image(image, 620, 620)
                rotated_img_cv = perform_object_detection(resized_image)
                rotated_img_cv = resize_image(rotated_img_cv, original_width, original_height)

                    # Convert the rotated image back to PIL format
                new_edited_img = rotated_img_cv

                    # Display the rotated image
                st.image(new_edited_img, caption="New image")   
                download()


else:
    for i, tasks in enumerate(results_list):

        if tasks == "rotate":
            # index = min(i, len(num_from_prommpt) - 1)
            # value = num_from_prommpt[index]


            def rotate_function(image, degree):
                if degree == 180:
                    output = cv2.rotate(image, cv2.ROTATE_180)

                elif degree == 90:
                    output = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

                elif degree == 270:
                    output = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

                else:
                    output = image

                return output
                
            img_cv = np.array(image)

            # Rotate the image
            if len(num_from_prommpt) == 1:
                index = min(i, len(num_from_prommpt) - 1)
                value = num_from_prommpt[index]
                rotated_img_cv = rotate_function(img_cv, value)
            
            elif len(num_from_prommpt) == 0 or len(num_from_prommpt) > 1:
                degree_value = st.selectbox('choose the degree to rotate',('90', '180', '270'))
                rotated_img_cv = rotate_function(img_cv, int(degree_value))

            # Convert the rotated image back to PIL format
            new_edited_img = Image.fromarray(rotated_img_cv)
            st.image(new_edited_img, caption="New image")
            file_name = "new1.jpg"
        
            new_edited_img.save(file_name)
            download()



        elif tasks == "blur":
            

            def stack_blur(image, blur_percentage):
            
                # Check if blur percentage is 0, return the original image
                if blur_percentage == 0:
                    return image

                # Calculate the blur factor based on the image size and blur percentage
                image_height, image_width, _ = image.shape
                blur_factor = int((max(image_height, image_width) / 100.0) * blur_percentage)

                # Apply the stack blur effect
                blurred_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                blurred_image = cv2.blur(blurred_image, (blur_factor, blur_factor))
                blurred_image = cv2.cvtColor(blurred_image, cv2.COLOR_RGB2BGR)

                return blurred_image
            

            img_cv = np.array(image)

            if len(num_from_prommpt) == 1:
                index = min(i, len(num_from_prommpt) - 1)
                value = num_from_prommpt[index]
                rotated_img_cv = stack_blur(img_cv, value)
            
            elif len(num_from_prommpt) == 0 or len(num_from_prommpt) > 1:
                blur_percent = int(st.slider('adjust the blur percentage', 0, 100))
                rotated_img_cv = stack_blur(img_cv, blur_percent)

            # Rotate the image
            #rotated_img_cv = stack_blur(img_cv, value)

            # Convert the rotated image back to PIL format
            new_edited_img = Image.fromarray(rotated_img_cv)

            # Display the rotated image
            st.image(new_edited_img, caption="New image")
            download()
        



        elif tasks == "brightness":
            

            # Prompt for brightness percentage

            def modify_brightness(image, brightness_percentage):
                hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

                h, s, v = cv2.split(hsv_image)

                brightness_factor = float(brightness_percentage) / 100.0
                v = np.clip(v * brightness_factor, 0, 255).astype(np.uint8)
                hsv_image = cv2.merge([h, s, v])

                modified_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
                return modified_image

            
            img_cv = np.array(image)

            if len(num_from_prommpt) == 1:
                    index = min(i, len(num_from_prommpt) - 1)
                    value = num_from_prommpt[index]
                    rotated_img_cv = modify_brightness(img_cv, value)
                
            else:
                brightness_value = int(st.slider('adjust the brightness percentage', 0, 200))
                rotated_img_cv = modify_brightness(img_cv, brightness_value)

            # Rotate the image
            #rotated_img_cv = modify_brightness(img_cv, value)

            # Convert the rotated image back to PIL format
            new_edited_img = Image.fromarray(rotated_img_cv)

            # Display the rotated image
            st.image(new_edited_img, caption="New image")  
            download()    




        elif tasks == "crop":
            
            def image_crop(image):
                height, width, channel= image.shape

                #first_num, second_num = ratio.split(":")

                a = width
                b = height

                new_height_a = st.slider('adjust the height_a', 0, int(height))
                new_height_b = st.slider('adjust the height_b', int(height), 1)
                new_width_a = st.slider('adjust the width_a', 0, int(width))
                new_width_b = st.slider('adjust the width_b', int(width), 1)

                edit_img = cv2.rectangle(image, (new_width_a,new_height_a) , (new_width_b, new_height_b), (255,255,255), 3)
                st.image(edit_img)
                #st.button('Ok')

                st.write("This is the output image. You can adjust the slide before download")

                if new_height_a >= new_height_b:
                    if new_width_a >= new_width_b:
                        ah = new_height_a
                        bh = new_height_b
                        aw = new_width_a
                        bw = new_width_b

                    else:
                        ah = new_height_a
                        bh = new_height_b
                        aw = new_width_b
                        bw = new_width_a


                else:
                    if new_width_a >= new_width_b:
                        ah = new_height_b
                        bh = new_height_a
                        aw = new_width_a
                        bw = new_width_b


                    else:
                        ah = new_height_b
                        bh = new_height_a
                        aw = new_width_b
                        bw = new_width_a


                crop_image = edit_img[bh: ah, bw: aw]


                return crop_image

            img_cv = np.array(image)
                
            # Rotate the image
            rotated_img_cv = image_crop(img_cv)

                # Convert the rotated image back to PIL format
            new_edited_img = Image.fromarray(rotated_img_cv)

                # Display the rotated image
            st.image(new_edited_img, caption="New image")   
            download()  
            


        elif tasks == "resize":
            def resize_function(image, percentage):
                width = int(image.shape[1] * percentage/100)
                height = int(image.shape[0] * percentage/100)
                new_size = (width, height)
                resized_img = cv2.resize(image, new_size, interpolation = cv2.INTER_AREA)
                return resized_img
            
            img_cv = np.array(image)

            if len(num_from_prommpt) == 1:
                index = min(i, len(num_from_prommpt) - 1)
                value = num_from_prommpt[index]
                rotated_img_cv = resize_function(img_cv, value)
           
            else:
                resize_value = st.slider('adjust the percentage to resize', 1, 200)
                rotated_img_cv = resize_function(img_cv, resize_value)
                
            
            new_edited_img = Image.fromarray(rotated_img_cv)

                # Display the rotated image
            st.image(new_edited_img, caption="New image")   
            download()  



        elif tasks == "detect":

            def resize_image(image, width, height):
                resized_image = image.resize((width, height))
                return resized_image
                       
            # Load the YOLOv5 model
            model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

            # Define a function to perform object detection on the uploaded image
            def perform_object_detection(image):
               
                results = model(image)

                annotated_image = Image.fromarray(results.render()[0])

                return annotated_image
            
            #img_cv = np.array(image)
            original_width, original_height = image.size

            resized_image = resize_image(image, 620, 620)
            rotated_img_cv = perform_object_detection(resized_image)
            rotated_img_cv = resize_image(rotated_img_cv, original_width, original_height)

                # Convert the rotated image back to PIL format
            new_edited_img = rotated_img_cv

                # Display the rotated image
            st.image(new_edited_img, caption="New image")   
            download()           





