Introduction
The Transformer Failure Prediction System is a deep learning-based predictive maintenance solution. It utilizes Convolutional Neural Networks (CNN) and Continuous Wavelet Transform (CWT) to predict real-time transformer failures, enhancing the reliability and stability of power grids.

System Requirements
Hardware:

Minimum 8GB RAM and a modern multi-core processor.
Reliable internet connection (for updates/cloud processing).
Current measurement devices for power transformers.
Data acquisition system to capture phase current waveforms.
Software:

Operating System: Windows, macOS, or Linux.
Python 3.7+ with libraries:
Streamlit
TensorFlow
Keras
NumPy
Matplotlib
Setup Instructions
Hardware Setup:

Install current transformers (CTs) on power lines and connect them to a data acquisition system (DAQ).
Configure DAQ to sample current waveforms at 10 kHz or higher and ensure synchronization with the operational frequency.
Use a signal processing unit to convert waveforms to scalogram images using a CWT algorithm.
Software Installation:

Install Python and required libraries:
Copy code
pip install streamlit tensorflow keras numpy matplotlib
Clone the project repository.
Running the Web Interface:

Open a terminal and navigate to the project directory.
Run the application:
arduino
Copy code
streamlit run app.py
Usage Instructions
User Authentication:

Login with valid credentials.
Model Selection:

Choose from available models and load the required files.
Uploading Scalogram Images:

Upload the images for analysis.
Running Analysis:

Click "Run Analysis" to process images and display results.
Displaying Results:

View combined scalogram images and prediction results on the interface.
Maintenance and Troubleshooting
Regular Maintenance:

Update software and libraries regularly.
Inspect hardware connections and clean components to prevent overheating.
Common Issues:

Web Interface Not Loading: Check for terminal errors, ensure dependencies are installed, and verify internet connection.
Incorrect Predictions: Confirm scalogram image quality and format. Ensure the model is correctly trained.
Best Practices
Enhance data augmentation for diverse training datasets.
Explore advanced CNN architectures and hyperparameters.
Implement continuous learning frameworks.
Use cloud-based solutions for scalability.
Develop an intuitive and secure user interface.
Contact Information
For support, contact Orugun David Modupe at orugundavid6@gmail.com.

This README provides an overview of the system's features and setup. For detailed information, refer to the user manual.
