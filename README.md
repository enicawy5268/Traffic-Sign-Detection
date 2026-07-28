# Traffic-Sign-Detection
🚦 Traffic Sign Recognition

📌 Project Overview

This project implements a computer vision pipeline using OpenCV to detect and segment traffic signs based on their color and shape.
The system applies color masking in the HSV space, shape analysis with contour approximation, and visualization through multi-window outputs. It is designed as part of an academic assignment to demonstrate traffic sign detection techniques.

🔧 Features
Color Detection: Extracts red, blue, and yellow regions using HSV thresholds.

Shape Analysis: Identifies common traffic sign shapes such as circles, triangles, rectangles, and octagons.

Segmentation: Generates masks to isolate traffic signs from the background.

Visualization:

- Six-window view showing each stage of segmentation.

- Left-right comparison view (original vs segmented sign).

Result Saving: Cropped traffic signs are automatically saved to the output directory.

Display Output:
<img width="885" height="610" alt="image" src="https://github.com/user-attachments/assets/0c22a45f-365d-43d4-b43f-acdd5bce607a" />
<img width="587" height="322" alt="image" src="https://github.com/user-attachments/assets/75c80434-2f1f-4888-80d3-ecd49f262e3e" />

