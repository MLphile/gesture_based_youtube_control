
# Controlling Youtube player with hand gestures
The goal of this project is to use an Artificial Neural Network to recognise a set of hand gestures and use those to interact with a YouTube player. Additionally, if the user is sleeping or has left, the player will get paused automatically.  Here is a demo!!!  
Below is a list of all the implemented interactions.    

Why Youtube player, you might ask? Well, it's popular, there is no need to install a software locally, you can find pretty much any kind of visual content and it's free (as long as you're ok with advertisement ;).  
But of course, you can implement the same technique to control a local media player or anything else; just make sure the application allows for keyboard shortcuts or has an API. 


## Intro
Gesture-based interfaces are systems that allow users to interact with them by using hand or other body components. These applications are getting more and more popular and have a wide range of use cases; for example in home automation, healthcare, automative, virtual reality, etc. If well designed, gesture-based interfaces feel more natural, intuitive and easier to learn.
## Approach
### Hand gestures
The approach used for hand gesture detection was highly inspired by Nikita Kiselov.  One of the advantages of this approach is that, you don't need to collect tons of images to train your model, since you rather use landmarks as model inputs.  
The worklow is as follows:  
* I extracted 2D coordinates from MediaPipe's hand detector.  This detector normally outputs 3D 20 landmarks, as shown in the image below. In contrast to Nikita, I further restricted the points to only wrist and tip coordinates. Wrist coordinates were then subtracted from the rest of the points. These new points were then flattened and normalized by the maximum absolute value. Also, I computed the distances between keypoints 4, 8 and 12. Those distances were also normalized by the distance between points 0 and 5. It's worth mentioning that only the left hand was considered in this project.  
* Both normalized coordinates and distances were then joined together to formed our feature space, then saved together with the target for subsequent training. Go to Usage section to see how to log data.
* Because of the preceding preprocessing steps and the simplicity of the data (13 features and 13 classes of approx. 30 samples each), I trained a simple artificial neural network. The architecture looks like this:
  
### Sleepness detection
For this feature, I took inspiration from this Adrian Rosebrock's blog.  The idea is to :
* Detect the face, using the frontal face detector from the Dlib library. This is how to install Dlib on Ubuntu and macOS and here on windows 10. 
* Pass the detected face into Dlib shape predictor to output facial landmarks. The pretrained model can be downloaded here (95.1 MB).
* From facial landmarks, extract eye landmarks and compute the so-called eye aspect ratio (EAR) to determine when the user is sleeping. User is considered to be sleeping if the EAR drops under a given threshold and stays under that value for at least a predefined number of consecutive frames.  
Alternatively, you can use MediaPipe's face mesh together with the EAR concept to achieve the same result. I've found MediaPipe to be more stable and robust (e.g., less sensitive to occlusion). The only problem is that it outputs 468 landmarks, which drastically slowed down the execution of my code. If you have a more powerful hardware, you should definitely give it a try. 
### Absence detection
The implementation of this feature was pretty straightforward:
* Detect the presence of a face using MediaPipe's face detector.
* If a face is not detected for at least a prefined number of consecutives frames, then the user is considered to be absent.
## Project Structure
```bash
 ┣━ 📂data
 ┃ ┣━ 📜check_data.ipynb
 ┃ ┣━ 📜gestures.csv
 ┃ ┣━ 📜label.csv
 ┃ ┗━ 📜player_state.json
 ┣━ 📂flask_app
 ┃ ┣━ 📂static
 ┃ ┃ ┣━ 📂css
 ┃ ┃ ┃ ┗━ 📜styles.css
 ┃ ┃ ┗━ 📂icons
 ┃ ┃ ┃ ┗━ 📜favicon-32x32.png
 ┃ ┣━ 📂templates
 ┃ ┃ ┗━ 📜demo.html
 ┃ ┣━ 📜app.py
 ┃ ┗━ 📜video_feed.py
 ┣━ 📂models
 ┃ ┣━ 📜model.pth
 ┃ ┣━ 📜model_architecture.py
 ┃ ┗━ 📜shape_predictor_68_face_landmarks.dat
 ┣━ 📜main.py
 ┣━ 📜requirements.txt
 ┣━ 📜train.ipynb
 ┗━ 📜utils.py
```

### main.py
For saving data and checking the output of models.

### utils.py
A collection of functions used in main.py.

### train.ipynb
For training and validating our artificial neural network.

### data/
Folder containing saved data (`gestures.csv`), general information about the saved data (`check_data.ipynb`) and gestures names (`label.csv`).  
The `player_state.json` is automatically generated and gives information whether the player is in pause or playing mode.

### models/
Contains the trained neural network (`model.pth`) and it's architecture(`model_architecture.py`) as well as the face landmarks predictor (`shape_predictor_68_face_landmarks.dat`)
### flask_app/
Contains important files for running the web app.

## Usage
NB: I'm using windows 10.
### Installing libraries
I suggest creating a virtual environment and installing the libraries there
```
cd project_folder_name
python -m venv your_virtual_env_name
your_virtual_env_name\Scripts\activate.bat
pip install -r requirements.txt  
```
### Saving data
Run `main.py`.  
Press 'r' on the keyboard to activate the logging mode. By pressing '0' to '9', data get saved in a csv file (see example below); whereby the first column represents the class labels (pressed keys) and the other columns are the normalized keypoints and distances. To save class labels extending from '10' to potentially '35', you can press alphabet keys (capital letters) from 'A' to 'Z', respectively.  
If you change the number of classes, make sure to correspondingly update the variable `n_classes` in `model_architecture.py` file.
### Training
For training the model, simply run the entire file `train.ipynb`.  
If you change the model architecture, make sure to correspondingly update the `model_architecture.py` file.
### Running the web app
```
cd flask_app
python app.py
```
You'll be provided with a link where the app is running (see image below).  
Go to that link and you should have something like this:
Copy paste a youtube video link in the input field and hit start.
Both the youtube video and your webcam video will load into the web page.  
Hand gestures are valid only when your hand is in the red box on your webcam video. This is to prevent unintentional interactions with the player (e.g. when scratching your face).
 

## Limitations
* In low light conditions, hand landmark predictions are less stable and thus the detected gestures. Same applies to face detectors, as the image gets less clear.
* The sleepness detection works well only when your face is frontal to the camera. Dlib's face detector expects a frontal face.
* No detection if you go far away from the web cam.
Please, let me know if you face other issues or have any question.  
All feedbacks on what to improve are welcome :) 

## References
* [MediaPipe](https://google.github.io/mediapipe/)
* [Dlib](http://dlib.net/)
* [Nikita Kiselov](https://github.com/kinivi)
* [Adrian Rosebrock](https://pyimagesearch.com/author/adrian/)
* [Artificial Intelligence in Gestural Interfaces](https://emerj.com/ai-sector-overviews/artificial-intelligence-in-gestural-interfaces/)
