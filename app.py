import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.applications import ResNet50V2 # type: ignore
import json
import numpy as np
from PIL import Image
import os

class_names = ['Coleus scutellarioides', 'Phyllanthus niruri', 'Corchorus olitorius',
'Momordica charantia', 'Euphorbia hirta', 'Curcuma longa',
'Carmona retusa', 'Senna alata', 'Mentha cordifolia Opiz',
'Capsicum frutescens', 'Hibiscus rosa-sinensis', 'Jatropha curcas',
'Ocimum basilicum', 'Nerium oleander', 'Pandanus amaryllifolius',
'Aloe barbadensis Miller', 'Lagerstroemia speciosa', 'Averrhoea bilimbi',
'Annona muricata', 'Citrus aurantiifolia', 'Premna odorata',
'Psidium guajava', 'Gliricidia sepium', 'Citrus sinensis',
'Mangifera indica', 'Citrus microcarpa', 'Impatiens balsamina',
'Arachis hypogaea', 'Tamarindus indica', 'Leucaena leucocephala',
'Ipomoea batatas', 'Manihot esculenta', 'Antidesma bunius',
'Citrus maxima', 'Vitex negundo', 'Moringa oleifera',
'Blumea balsamifera', 'Origanum vulgare', 'Pepromia pellucida',
'Centella asiatica', 'Acer palmatum', 'Aesculus chinensis',
'Albizia julibrissin', 'Aloevera', 'Amruthaballi', 'Arali',
'Astma weed', 'Bamboo', 'Beans', 'Betel', 'Bhrami',
'Camptotheca acuminata', 'Castor', 'Catharanthus', 'Cedrus deodara',
'Celtis sinensis', 'Cinnamomum camphora', 'Citron lime', 'Coffee',
'Coriender', 'Curry', 'Doddpathre', 'Ekka', 'Elaeocarpus decipiens',
'Eucalyptus', 'Euonymus japonicus', 'Flowering cherry', 'Ginger',
'Ginkgo biloba', 'Globe Amarnath', 'Guava', 'Henna', 'Hibiscus',
'Honge', 'Insulin', 'Jackfruit', 'Koelreuteria paniculata',
'Lagerstroemia indica', 'Lemon', 'Liquidambar formosana',
'Liriodendron chinense', 'Loropetalum chinense var. rubrum',
'Magnolia grandiflora', 'Magnolia liliflora', 'Malushalliana',
'Mango', 'Marigold', 'Michelia chapensis', 'Mint', 'Neem', 'Nelavembu',
'Onion', 'Osmanthus fragrans', 'Palak', 'Papaya',
'Photinia serratifolia', 'Platanus', 'Populus', 'Prunus persica',
'Pumpkin', 'Rose', 'Salix babylonica', 'Sapindus saponaria',
'Seethapala', 'Styphnolobium japonicum', 'Tamarind', 'Triadica sebifera',
'Tulsi', 'Zelkova serrata', 'ashoka']

# Load the trained model
def load_model_flexible(model_path):

    if not os.path.exists(model_path):
        print(f"Error: The path {model_path} does not exist.")
        return None
    
    # Method 3: Try reconstructing the model and loading weights
    try:
        base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        predictions = tf.keras.layers.Dense(110, activation='softmax')(x)  # Adjust the number of classes as needed
        model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
        model.load_weights(model_path)
        print(f"Successfully reconstructed model and loaded weights")
        return model
    except Exception as e:
        print(f"Failed to reconstruct model and load weights: {str(e)}")
    
    print("All loading methods failed.")
    return None

# Usage
model_path = 'Leaf Classification models/leaf_detection_model.h5'  # Replace with your actual model file path
model = load_model_flexible(model_path)

if model:
    # If the model loaded successfully, you can compile it and use it
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # You can now use the model for predictions or further training
else:
    print("Failed to load the model.")


with open('leaf_info.json', 'r') as json_file:
    leaf_info = json.load(json_file)

def predict_img(path):
    img = image.load_img(path, target_size=(224, 224))
    img_arr = image.img_to_array(img)
    img_arr_expnd = np.expand_dims(img_arr, axis=0)
    img = tf.keras.applications.resnet_v2.preprocess_input(img_arr_expnd)

    pred = model.predict(img)
    result = class_names[np.argmax(pred)]
    
    

    return result, pred

# Replace with actual class names
disease_classes = ['Healthy', 'Powder', 'Rust']  # Replace with actual disease names


def disease_load_model_flexible(model_path):

    if not os.path.exists(model_path):
        print(f"Error: The path {model_path} does not exist.")
        return None
        
    # Method 3: Try reconstructing the model and loading weights
    try:
        base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(128,activation='relu')(x)
        predictions = tf.keras.layers.Dense(3, activation='softmax')(x)  # Adjust the number of classes as needed
        model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
        model.load_weights(model_path)
        print(f"Successfully reconstructed model and loaded weights")
        return model
    except Exception as e:
        print(f"Failed to reconstruct model and load weights: {str(e)}")
        
    print("All loading methods failed.")
    return None

disease_model_path = 'Leaf Disease Detection models/disease_recognition_resnet.h5'  # Replace with your actual model file path
disease_model = disease_load_model_flexible(disease_model_path)

# Define a function to preprocess the image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))  # Adjust target size as needed
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize to [0, 1]
    return img

# Define a function to make predictions
def make_prediction(image_path):
    img = preprocess_image(image_path)


    # Predict the disease
    disease_pred = disease_model.predict(img)
    disease_class = disease_classes[np.argmax(disease_pred)]

    return disease_class, disease_pred


st.title("Leaf Classification App")

# Allow users to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Save the uploaded image to a temporary file
    temp_file_path = "temp_image.png"
    img.save(temp_file_path)
    
    # Make prediction
    result, predictions = predict_img(temp_file_path)

    if st.button('Show Prediction'):
        # Display the prediction
        st.write(f"**Predicted class:** {result}")
        st.write(f"")
        
        if result in leaf_info:
            st.write(f"Information: {leaf_info[result]}")
        else:
            st.write("No information available for this leaf.")
            
            
            
        # Sort probabilities and get top 10 classes
        top_k = 10
        top_k_indices = np.argsort(predictions[0])[-top_k:]
        top_k_indices = top_k_indices[::-1]
        top_k_labels = np.array(class_names)[top_k_indices]
        top_k_probabilities = np.array(predictions[0])[top_k_indices]
            
            
        st.subheader("Top 10 Class Probabilities:")
        probability_table = {
            "Class": top_k_labels,
            "Probability": top_k_probabilities
        }
        st.table(probability_table)
        
        
        disease_class, disease_pred = make_prediction(uploaded_file)

                
        st.write(f"**Detected Disease:** {disease_class}")
            
        st.subheader("Disease Probabilities:")
        probability_table = {
            "Class": disease_classes,
            "Probability": disease_pred[0]
        }
        st.table(probability_table)
    