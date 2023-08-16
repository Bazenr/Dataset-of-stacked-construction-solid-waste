# **Object-detection-code/Dataset-generation-code/Dataset of construction solid waste**
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
|**Content**|**Location / Link**|
|:--------|:-------------|
|**Dataset samples**|Dataset-of-stacked-construction-solid-waste/**Dataset**|
|**Dataset generation code**|Dataset-of-stacked-construction-solid-waste/**Rapid_generation**|
|**Object detection code**|Dataset-of-stacked-construction-solid-waste/**Experiment_code**|
|**Sourse dataset link**|https://www.kaggle.com/datasets/bazenr/stacked-construction-solid-waste|

The dataset contain **4 types** of construction solid waste: **concrete**, **brick**, **wood** and **rubber**. Each document countains 25 sample images and label files.

**Dataset samples:**

<img width="355" alt="308400 000000-" src="https://github.com/Bazenr/Dataset-of-stacked-construction-solid-waste/assets/81945216/3b5611b8-1e3e-45d1-978b-d9d1a1e92b50">

**"A"** represent simple work condition, and **"B"** represent complex work condition.

**"auto"** means the dataset was **automatically labeled** using OpenCV according to height images. And the labels are correct, since all objects in one image must belong to the **same category**. The folder **"color"**, **"height"** and **"json"** represent **RGB**, **height** images and **label** files.

**"cp"** means the dataset was **automatically copy-pasted** according to the data in corresponding **"auto"** dataset.

**"manual"** means the dataset was **labeled manually** by us.

**"test"** means the dataset was **labeled manually** by us for **model test**. The images never appear in other files.
