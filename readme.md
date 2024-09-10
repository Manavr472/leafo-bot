
---

# Leaf Classification and Disease Detection

This project focuses on classifying different types of leaves and detecting diseases in them. It uses deep learning models trained on specific datasets to perform these tasks efficiently.

## Project Structure

```bash
├── Leaf Classification models     # Contains pre-trained models for leaf species classification
├── Leaf Disease Detection models  # Contains pre-trained models for leaf disease detection
├── notebooks                      # Jupyter notebooks for training, evaluation, and experiments
├── app.py                         # Main application script to run the Streamlit web app
├── leaf_info.json                 # JSON file containing one-liners about the usage and information for each leaf
├── .gitattributes                 # Git configuration file
```

## Setup Instructions

### 1. Install Dependencies

Before running the project, you need to install all required dependencies. A `requirements.txt` file should be present in the project directory. Use the following command to install them:

```bash
pip install -r requirements.txt
```

### 2. Running the Application

The project is built using **Streamlit** for the web interface. To run the application, use the following command:

```bash
streamlit run app.py
```

This will launch the web app in your browser, where you can classify leaves and detect diseases by uploading leaf images.

### 3. Models and Data

- **Leaf Classification Models**: These models are used to classify various species of leaves.
- **Leaf Disease Detection Models**: These models are used to detect diseases affecting different types of leaves.

The model files are stored in the respective directories: `Leaf Classification models` and `Leaf Disease Detection models`.

## How It Works

1. **Leaf Classification**: Upload a leaf image, and the model will predict the species of the leaf.
2. **Disease Detection**: Once the leaf species is identified, the corresponding disease detection model will analyze the leaf for any potential diseases.
3. **Leaf Information**: The app also displays information about the predicted leaf species, which is stored in the `leaf_info.json` file.

## Requirements

- Python 3.x
- TensorFlow/Keras
- Streamlit
- NumPy
- OpenCV (for image preprocessing)
- Pillow (for handling images)

Make sure you have these dependencies installed by using the `requirements.txt`.

## Example Usage

1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```

2. Navigate to the project directory:
   ```bash
   cd project-directory
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

5. Upload a leaf image in the web interface, and the app will display the classification result along with leaf information.

---