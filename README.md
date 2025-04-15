# Color Classification with Neural Network

A lightweight embedded system for **classifying seven distinct colors** based on analog light intensity measurements from RGB-filtered photoresistors. Classification is performed using a simple neural network (MLP) running on an **NUC140 microcontroller**.

---

## Contents
- [Hardware Platform](#hardware-platform)
- [Circuit Design](#circuit-design)
- [System Architecture](#system-architecture)
- [Data Acquisition & Preprocessing](#data-acquisition--preprocessing)
- [Neural Network Model](#neural-network-model)
- [Implementation Workflow](#implementation-workflow)

---

## Hardware Platform

### NUC140 V2.0 Development Board  
[Official Product Page – Nuvoton](https://www.nuvoton.com/products/microcontrollers/arm-cortex-m0-mcus/nuc140-240-connectivity-series/?__locale=zh_TW)

<img src="https://github.com/user-attachments/assets/f774884e-337a-4af0-b76f-806cf17f2b3a" width="500"/>

---

## Circuit Design

**Main Components:**
- Photoresistors with color filters (Red, Green, Blue)
- Resistors (for voltage division)
- Status LED
- PL2302 USB-to-Serial Converter

<img src="https://github.com/user-attachments/assets/ab2914f6-f724-44c2-b110-3a1aacdc9ba8" width="500"/>

---

## System Architecture

### Forward Propagation
- Normalized RGB input flows through the MLP to produce class scores.

### Backward Propagation
- Error signal is computed and propagated backward to update weights.

### Training Process
- Normalize inputs → initialize parameters → train over multiple epochs → monitor performance.

### Real-Time Inference
- Live sensor readings are normalized and fed into the trained model to predict the current color class.

---

## Data Acquisition & Preprocessing

### Sensor and Sampling
- RGB light intensities are measured using photoresistors with corresponding color filters.
- Analog signals are digitized via NUC140’s ADC.

### Dataset
- 30 samples per color
- 7 color categories
- Serial data captured via PuTTY terminal

### Preprocessing Pipeline
- Compute feature-wise mean and standard deviation
- Normalize dataset before training

---

## Neural Network Model

### Architecture
- Simple MLP with one or more hidden layers

### Activation & Output
- Sigmoid activation on output layer for probability distribution across 7 classes

### Learning Mechanism
- Error is minimized using backpropagation and weight updates based on gradient descent

<img src="https://github.com/user-attachments/assets/cafc43ed-7527-4d56-b658-cb099d3997fb" width="300"/>

---

## Implementation Workflow

### 1. Sensor Input & ADC Conversion
- RGB light intensity is converted into digital signals
- Triggered by `ADC_ADF_INT` interrupt

### 2. Data Communication (UART)
- Transmit ADC results to PC during training phase
- Use serial terminal (e.g., PuTTY) for real-time logging

### 3. On-Device Inference
- Once trained, model is deployed to the MCU for standalone operation
- Color classification runs in real time
