# Ball Tracking and Classification Project

## Overview

This project involves tracking a ball's position in a video and classifying the ball's state using a neural network model. The classification results are used to determine the shot to be played in a game scenario.

## Features

- Ball tracking using OpenCV template matching.
- Ball state classification with a neural network model.
- Automated shot determination and execution based on ball position and state.

## Installation

To set up the project, follow these steps:

1. Clone the repository:

```
git clone https://github.com/kukr/Transforming-Gameplay-Through-Vision-Automation.git
```

2. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the ball tracking and classification:

1. Execute the `automation.py` script:

```
python automation.py
```

2. Adjust the trackbars in the OpenCV window to set the area of interest.

3. The script will track the ball, classify its state, and simulate the appropriate shot.

## Neural Network Model

The neural network model is trained to classify the ball's state into three categories: Left, Not Active, and Right. The model is defined and trained in the `Our_NeuralNetwork_Model.ipynb` Jupyter notebook.

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
