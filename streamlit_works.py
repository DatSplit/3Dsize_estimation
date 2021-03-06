import streamlit as st
import numpy as np
import mediapipe as mp
import cv2
from PIL import Image



# Read logo
logo = cv2.imread('datalier_logo.png')
logo = cv2.cvtColor(logo,cv2.COLOR_BGR2RGB)

# Display logo
st.image(logo)

st.subheader('Object identification and dimension measurements.',anchor="Object identification and dimension measurements.")

# Add selection box for objects
option = st.selectbox(
'Which object would you like to identify and measure?',
('Cup', 'Chair', 'Shoe', 'Camera'))

# Add number input box
distance = st.number_input('What is the distance to the object in (mm)?')
if(distance == 0):
    st.error('Please enter a number.')

# Camera parameters of image
sensor_height_y = 3.6 #mm
focal_length_camera = 2.65 #mm
sensor_height_x = 3.6
x_width = 1280
y_height = 720

# Add box to upload image to be identified and measured
image2 = st.file_uploader('Upload an image containing a ' + str(option), help="Upload image containing a Cup with size", type=['png','jpg'])


mp_drawing = mp.solutions.drawing_utils
mp_objectron = mp.solutions.objectron

logo = cv2.imread("logo_wit.png")
logo = cv2.resize(logo, (264, 57),
                  interpolation=cv2.INTER_CUBIC)  # 528,114

# Convert image to grey and create mask by thresholding.
img2gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
logo = cv2.cvtColor(logo, cv2.COLOR_RGB2BGR)
# For static images:
with mp_objectron.Objectron(static_image_mode=True,
                            max_num_objects=5,
                            min_detection_confidence=0.3,
                            model_name=str(option),
                            ) as objectron:
    if (image2 != None):

        image = Image.open(image2)
        image = image.resize((1280, 720))
        img_array = np.array(image)
        # Convert the BGR image to RGB and process it with MediaPipe Objectron.


        results = objectron.process(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))

        # Draw box landmarks.
        if not results.detected_objects:
            print(f'No box landmarks detected')
            st.write('No ' + str(option) + ' detected.')
        else:

            annotated_image = img_array.copy()


            for detected_object in results.detected_objects:

              mp_drawing.draw_landmarks(
                  annotated_image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)



              if(str(option) == "Cup" or "Camera" or "Chair"):

                  # height,length, and width calculation of the object (Cup, Camera, or Chair).
                  # See the jupyter notebooks for explanations of the calculations.
                  height_shoe = abs(detected_object.landmarks_2d.landmark[2].y - detected_object.landmarks_2d.landmark[4].y)
                  height_shoe_pixels = height_shoe * y_height
                  height_shoe_cm = ((distance * height_shoe_pixels * sensor_height_y) / (focal_length_camera * y_height))

                  pixels_per_metric_y = height_shoe_pixels / height_shoe_cm

                  length_shoe = abs(detected_object.landmarks_2d.landmark[2].x - detected_object.landmarks_2d.landmark[6].x)
                  length_shoe_pixels = length_shoe * x_width
                  length_shoe_cm = ((distance * length_shoe_pixels * sensor_height_x) / (focal_length_camera * x_width))


                  width_shoe = abs(detected_object.landmarks_2d.landmark[2].y - detected_object.landmarks_2d.landmark[1].y)

                  width_shoe_cm = (width_shoe * y_height) / pixels_per_metric_y

              if(str(option) == "Shoe"):


                  # Different landmarks are required for the length, width, and height of the shoe, due to the natural orientation of the shoe.
                  # See the jupyter notebooks for explanations of the calculations.
                  height_shoe = abs(
                      detected_object.landmarks_2d.landmark[5].y - detected_object.landmarks_2d.landmark[7].y)
                  height_shoe_pixels = height_shoe * y_height
                  height_shoe_cm = (
                              (distance * height_shoe_pixels * sensor_height_y) / (focal_length_camera * y_height))

                  pixels_per_metric_y = height_shoe_pixels / height_shoe_cm

                  length_shoe = abs(
                      detected_object.landmarks_2d.landmark[6].x - detected_object.landmarks_2d.landmark[7].x)
                  length_shoe_pixels = length_shoe * x_width
                  length_shoe_cm = ((distance * length_shoe_pixels * sensor_height_x) / (focal_length_camera * x_width))


                  width_shoe = abs(
                      detected_object.landmarks_2d.landmark[2].y - detected_object.landmarks_2d.landmark[6].y)

                  width_shoe_cm = (width_shoe * y_height) / pixels_per_metric_y

              annotated_image[665:, :] = [
                  14, 49, 88  #BGR -> RGB
              ]  # Create blue box at bottom of the videocapture

              # Find region of interest for the logo
              roi = annotated_image[-57 -630:-630, -264 - 30:-30]
              roi[np.where(mask)] = 0
              roi += logo

              # Add text about the length, width, and height.
              cv2.putText(
                  annotated_image, "Lengte: " + "{:.1f}".format((length_shoe_cm/10)) +
                         ' cm' + "  " + "Breedte: " + "{:.1f}".format(
                      (width_shoe_cm/10)) + ' cm'
                                             "  " + "Hoogte: " + "{:.1f}".format(
                      (height_shoe_cm/10)) + ' cm', (70,700), 2,
                  1.2, (255, 255, 255), 2)
              # Display the annotated image
              st.image(annotated_image)






