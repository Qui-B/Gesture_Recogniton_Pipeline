---
title: Project Description

---

#TODO tl;tr, Benchmark 

# Project Description
This repo contains the src files and builds for an Adobe Acrobat Plugin, created by using the [Adobe Acrobat SDK 2021](https://opensource.adobe.com/dc-acrobat-sdk-docs/acrobatsdk). The Plugin enables motion-based hand gestures for navigating through documents. 
For the use of the Plugin a [Recognition Pipeline](https://github.com/Qui-B/Gesture_Recognition_Pipeline) is requiered, which is also included in the download build.
# Getting started
### Windows 10/11
1. Download the lastest [release](https://github.com/Qui-BGesture_Recognition_Acrobat_Plugin/releases)
2. Open cmd navigate inside download-folder\recognition_system
3. Create the environment and built it by running:
```
python -m venv .venv && .venv\Scripts\activate && pip install -r src\requirements.txt
```
In case also GPU support is needed for the Classificationnetwork (GraphTCN) run the following command inside the environment:
```
pip install torch==2.5.1+cu121 `
    torchvision==0.20.1+cu121 `
    torchaudio==2.5.1+cu121 `
    --extra-index-url https://download.pytorch.org/whl/cu121
```
**However**, as the bottleneck of the architecture (Mediapipe as pipeline component) solely runs on the CPU, this version only comes with a **small performance benefit in  the best case**.

4. Put the downloaded folder in the Acrobat plugins folder.  
Currently the default path is:
```
\Program Files\Adobe\Acrobat DC\Acrobat\plug_ins
```
6. Start adobe acrobat reader 
7. Navigate to 'Menu (top left corner) -> Allgemein'
8. Untick the Option "Use only certified plugins" on the bottom of the Menu.
9. Navigate to 'Menu -> Plug-ins -> Gesture Recognition' and click on the desired starting option:
> - Debug: Displays bulky debug information used for development
> - Window Only: Displays the console output of the recognition system in a cmd window and the next Frame to be analysed in a seperate window.
>  >I. Yep, it's a video stream.
>  >  >II. And yep, i recommend using the "Window Only" option if you want to feel the power.
>  >  >  >III.And yep, nested block comments are bad.
>  >  >  >  >IV. But...i love writing block comments, especially nested ones.
>  >  >  >  >
>  >V. And yes Iwtyo.
>  >
> - Normal: No debug information
> 
//TODO vlt anpassenXD