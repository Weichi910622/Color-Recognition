# Color-Recognition

## Content
* [Device](#device)
* [Physical Circuit](#physical-circuit-setup)
* [Operating Principle](#operating-principle)
* [Data preprocessing](#data-collection-and-preprocessing)
* [Machine Learning](#machine-learning)
* [STEPS](#steps)

## Device

### [NUC140 V2.0 Development Board](https://www.nuvoton.com/products/microcontrollers/arm-cortex-m0-mcus/nuc140-240-connectivity-series/?__locale=zh_TW)
<img src="https://github.com/user-attachments/assets/f774884e-337a-4af0-b76f-806cf17f2b3a" width="500"/>

## Physical Circuit Setup
<img src="https://github.com/user-attachments/assets/ab2914f6-f724-44c2-b110-3a1aacdc9ba8" width="500"/>

## Operating Principle

### 1 . Forward Propagation➡

### 2. Backward Propagation⬅

### 3. Training Process

- Load and normalize data, initialize weights.
- Loop through training cycles with forward and backward passes.
- Track accuracy and error every few cycles.

### 4. Real-Time Prediction

- Once trained, the network switches to live mode.
- Sensor reads RGB → normalize → run prediction → display color result.

## Data Collection and Preprocessing

- Used three photoresistors with RGB filter papers for ADC conversion.
- Collected `30 samples` for each of the `7 color classes`.
- Data was printed and stored via Putty.

### Preprocessing Steps

- Compute mean and standard deviation for each feature.
- Normalize data to improve learning efficiency.

## Machine Learning
### With simple MLP

### **Forward Propagation**:
    - RGB sensor data is normalized and fed into the network.
    - Data flows through hidden layers to the output layer, where each neuron calculates a weighted sum and applies an activation function.
    - The Sigmoid function maps values to a range between 0 and 1, representing the probability of each color class.

### **Backward Propagation**:
    - Compute the error between prediction and target.
    - Propagate the error backward from output to hidden layers.
    - Adjust weights based on the calculated error.
![image](https://github.com/user-attachments/assets/cafc43ed-7527-4d56-b658-cb099d3997fb)


## Steps

### 1. Sensor Input via ADC(Collect data and predict)

- The RGB sensor (three photoresistors with color filters) captures light intensity.
- The NUC140’s built-in ADC converts analog signals into digital values (one for each R, G, B channel).
- A complete scan is triggered and controlled via interrupt flag (`ADC_ADF_INT`).

### 2. Data Transmission via UART

- ADC results are sent to the PC via UART (serial communication).
- During data collection, the PC receives and logs sensor readings using tools like PuTTY.
- After training, the MCU operates independently in prediction mode, displaying results locally.

### 3. Neural Network Inference (MLP)





