# Traffic Light Detection using Neural Networks


This repository contains code for a traffic light detection project using neural networks. The project aims to detect traffic lights with red and green lights facing towards the observer. It involves detecting potential traffic light coordinates in images, creating crops of these regions, and using a neural network to classify whether the crop contains a valid traffic light.
## Introduction

Traffic light detection is a crucial task for autonomous vehicles and smart traffic systems. This project demonstrates the process of detecting potential traffic light regions in images, creating crops of these regions, and classifying them using a neural network. The project aims to identify valid traffic lights with red and green lights directed towards the observer.

### Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.10 or later

- Required packages (specified in part_2/requirements.txt)


## Results

The project produces several results, including:

- Potential traffic light coordinates detected in images.
- Crops of potential traffic light regions.
- Classification results of valid traffic light crops.


## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/roeebenezra/TFL-detection-Mobileye.git
   ```

2. Install the required packages:

   ```bash
   pip install -r part_2/requirements.txt
   ```

## Usage

1. Prepare your dataset with annotated traffic light coordinates.
2. Configure the data paths and settings in the relevant scripts (`main.py`, `crops.py`, etc.).

3. View the generated results in the `attention_results` folder.

