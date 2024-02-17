**Dataset-of-stacked-construction-solid-waste**

Stacked C&amp;D waste dataset
The dataset contain 4 types of construction solid waste: **concrete**, **brick**, **wood** and **rubber**. Each document countains 25 sample images and label files.

Samples:

<img width="355" src="https://github.com/Bazenr/Dataset-of-stacked-construction-solid-waste/blob/master/data-original.png">

**"A"** represent simple work condition, and **"B"** represent complex work condition.

**"auto"** means the dataset was **automatically labeled** using OpenCV according to height images. And the labels are correct, since all objects in one image must belong to the **same category**. The folder **"color"**, **"height"** and **"json"** represent **RGB**, **height** images and **label** files.

**"cp"** means the dataset was **automatically copy-pasted** according to the data in corresponding **"auto"** dataset.

**"manual"** means the dataset was **labeled manually** by us.

**"test"** means the dataset was **labeled manually** by us for **model test**. The images never appear in other files.

Sourse dataset download link:
  https://www.kaggle.com/datasets/bazenr/stacked-construction-solid-waste
