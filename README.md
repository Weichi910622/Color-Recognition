# Color Classification with Neural Network

This project implements a lightweight embedded system that classifies seven different colors using light intensity readings from photoresistors covered with red, green, and blue filter sheets. A simple Multi-Layer Perceptron (MLP) neural network is used as the classification model, running on the **NUC140 microcontroller**.

---

## Table of Contents
- [Hardware Platform](#hardware-platform)
- [Circuit Design](#circuit-design)
- [System Architecture](#system-architecture)
- [Data Collection & Preprocessing](#data-collection--preprocessing)
- [Neural Network Model](#neural-network-model)
- [Implementation Steps](#implementation-steps)

---

## Hardware Platform

### NUC140 V2.0 Development Board  
[Official Product Page – Nuvoton](https://www.nuvoton.com/products/microcontrollers/arm-cortex-m0-mcus/nuc140-240-connectivity-series/?__locale=zh_TW)

<img src="https://github.com/user-attachments/assets/f774884e-337a-4af0-b76f-806cf17f2b3a" width="500"/>

---

## Circuit Design

**Main Components:**
- Photoresistors with red, green, and blue filters
- Voltage divider resistors
- Illumination LED (acts as active light source for reflection)
- PL2302 USB to UART module

**Sensing Design:**
- An LED provides active lighting and shines on the surface of the object being measured.
- The reflected light passes through a colored filter (glass or film) and reaches the corresponding photoresistor (R/G/B).
- Each photoresistor detects the reflected light intensity and converts it to an analog voltage.
- The voltage is divided and sent to the microcontroller’s ADC for digital conversion.

<img src="https://github.com/user-attachments/assets/ab2914f6-f724-44c2-b110-3a1aacdc9ba8" width="500"/>

---

## System Architecture

### Forward Propagation
- Normalized RGB input is passed through the MLP to calculate the prediction probabilities.

### Backward Propagation
- The error between the prediction and actual target is computed and propagated backward to update the weights.

### Training Flow
- Normalize input → initialize parameters → iterate training → monitor accuracy and loss

### Real-Time Inference
- Read RGB input → normalize → feed into MLP → display color classification result

---

## Data Collection & Preprocessing

### Sensing and Sampling
- RGB-filtered photoresistors measure reflected light intensity
- The analog signal is converted via the NUC140's ADC

### Dataset
- 30 samples per color
- 7 total color categories
- Data logged through serial interface using PuTTY

### Preprocessing Steps
- Compute mean and standard deviation per feature
- Normalize all data to improve model performance

---

## Neural Network Model

### Architecture
- A basic Multi-Layer Perceptron (MLP) with input, hidden, and output layers

### Activation and Output
- The output layer uses the Sigmoid function to estimate class probabilities

### Learning Process
- Uses backpropagation to compute gradients and updates weights via gradient descent

<img src="https://github.com/user-attachments/assets/cafc43ed-7527-4d56-b658-cb099d3997fb" width="300"/>

---

## Implementation Steps

### 1. Sensor Input and ADC Conversion
- RGB sensor reads analog voltages corresponding to light intensities
- `ADC_ADF_INT` interrupt flag is used to trigger the conversion process

### 2. UART Data Transmission
- During training, ADC results are sent to a PC via UART
- Data is captured using PuTTY or other terminal software

### 3. On-Chip Real-Time Inference
- After training, the model is deployed on the MCU
- The system runs standalone and performs real-time color prediction
