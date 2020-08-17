# carsim
Teleoperation of a Mobile robot under communication delays using donkeycar layout.

# Donkeycar

[Donkey Car](http://www.donkeycar.com) is an open source DIY self driving platform for small scale RC cars. You can interact with the Donkey environment using the familiar OpenAI gym like interface. The code was modified from [Tawn Kramer sdsandbox repo](https://github.com/tawnkramer/sdsandbox).

<img src="https://flyyufelix.github.io/img/ddqn_demo.gif" width="500">
<img src="https://i.imgur.com/6tU0elU.png" width="500">

## Usage

### Download Donkey Unity Environment
The Donkey Car simulator is created with Unity. There are 3 Unity scenes available (created by Tawn Kramer) for training now, which are generated roads, warehouse, and Sparkfun AVC. We have to either build the Donkey Car Unity environment ourselves (need to install Unity) or download the pre-built environment executables below:

Linux: [donkey.x86_64](https://drive.google.com/file/d/1p5Sn27o7YJC2SUBatCfUSlt9t-8xatDw/view?usp=sharing) | MacOS: [donkey.app](https://drive.google.com/drive/folders/1qfFkxlBy-nST3qcJzSQboVpIquzPHmsL?usp=sharing)

Then place the executable inside the `carsim` folder. 

### Install Donkeycar

First, we have to install `donkey_gym` python package, which extends the OpenAI gym class to allow ML developers to interact with Donkey environment using the familiar OpenAI gym like interface. 

To install the package, do the following guide:
Linux: [donkey install](https://docs.donkeycar.com/guide/install_software/)

## Dependencies

* [python 3.4+ 64 bit](https://www.python.org/)
* [pillow](https://python-pillow.org/)
* [opencv-python](https://pypi.org/project/opencv-python/)
* [python-socketio](https://pypi.python.org/pypi/python-socketio)
* [flask](https://pypi.python.org/pypi/Flask)
* [eventlet](https://pypi.python.org/pypi/eventlet)
* [Unity 5.5+](https://unity3d.com/get-unity/download)

