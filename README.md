
<h1 align="center">
  <br>
  <a href="#"><img src="https://raw.githubusercontent.com/rizkynat/FaceDetection/master/frontend/assets/logo.png" alt="FDetect" width="200"></a>
  <br>
  FDetect
  <br>
</h1>

<h4 align="center">Just click and get result of Face Detection.</h4>


<p align="center">
  <a href="#key-features">Features</a> •  
  <a href="#requirements">Requirements</a> •
  <a href="#ipcam-configuration">IPCam Configuration</a> •
  <a href="#instalation">Instalation</a> •
  <a href="#credits">Credits</a> •
  <a href="#colabolator">Colabolator</a>
</p>

<img src="https://raw.githubusercontent.com/rizkynat/FaceDetection/master/frontend/assets/demo_fdetect.gif" alt="My Project GIF" style="display: block;
  margin-left: auto;
  margin-right: auto;
  width: 100%;">
## Key Features

* Face Detection
  - HOG model
  - Haarcascade model
* Visualization
  - Data realtime total face in line chart
* Show total face has been detection
* Input camera from source has connect in computer
* Input source from IPCam using RTSP protocol
### Requirements
* Ezviz IPCam (You can use another IPCam product like Hikvision)
* Python 3.3+ or Python 2.7
* Windows
* Anaconda

### IPCam Configuration
##### Using Cable LAN
1. Connect your rj45/lan cable to your pc like this </br>
    <img src="https://raw.githubusercontent.com/rizkynat/FaceDetection/master/frontend/assets/img1.jpeg" alt="FDetect" width="600">
2. Type **"ipconfig"** in cmd and look ip address for adapter lan to check whether the cable is connected to the PC
3. Install ezviz studio app from this <a href="https://mfs.ezvizlife.com/EzvizStudioSetups.exe?ver=44195&_gl=1*81fpvs*_ga*MTc1NTE1MjY0MS4xNjkwOTYxNDA4*_ga_GFXNRVT2BW*MTY5MDk2MTQwOC4xLjAuMTY5MDk2MTQwOC42MC4wLjA.">link</a>
4. Sign up your account and see below **"all devices"** to expand
5. Look your display for IPCam and click **Network** to get ip address
6. To test your camera use **VLC media player** and click **Open Stream Network**

##### Wifi Router
1. Install ezviz from **play store/app store**
2. Follow the intructions from the app until you can connect to IPCam and see your IPCam's video in screen
3. If IPCam has connected, open ezviz studio
4. Look your display for IPCam and click **Network** to get ip address

##### How to match in IP address and RTSP Protocol
IP Cam uses the url format
```bash
rtsp://admin:password@ipaddress:554  
```
</br>

> **Note**
> the configuration above is not the main one and it must depend on the product from the IPCam that you are using.

### Installation
##### Basic configuration
</br>
install dlib and cmake library with follow <a href="https://medium.com/analytics-vidhya/how-to-install-dlib-library-for-python-in-windows-10-57348ba1117f">this</a>
if you using anaconda just type in anaconda's terminal:

```bash
conda install -c conda-forge dlib
conda install -c "conda-forge/label/cf201901" dlib
conda install -c "conda-forge/label/cf202003" dlib
```

##### Download
</br>

```bash
# Clone this repository
$ git clone 

# Go into the repository
$ cd 

# Install dependencies
$ pip install
- face_recognition
- opencv
- flask

# Run the app
$ python main.py

```
##### Run app
Open file demo.html in browser

> **Note**
> when the app crush, check your network has connected with IPCam.


### Credits

This software uses the following open source packages:

- [OpenCV](https://opencv.org/)
- [FaceRecognition](https://github.com/ageitgey/face_recognition)
- [DLib](http://dlib.net/ )
- [ffmpeg](https://www.ffmpeg.org/)



### Colabolator

[Mam syefrida](https://github.com/syefrida)

