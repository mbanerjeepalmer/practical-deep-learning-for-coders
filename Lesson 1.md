# Lesson 1

 https://www.kaggle.com/code/jhoward/is-it-a-bird-creating-a-model-from-your-own-data

Apparently Jeremy's original notebook is wrong. Update: https://www.kaggle.com/code/limwaijian/is-it-a-bird-more-than-2-categories-model


What mystery. Particularly when running this in a notebook environment, you have no idea what half of this stuff is.
```python
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path, bs=32)

dls.show_batch(max_n=12)
```

Especially because we have this menace:
```python
from fastai.vision.all import *
```

## Fine tune
![](assets/Pasted%20image%2020241228125803.png)


Doesn't really work much of the time. But if I pick something from the training set it works.

![](assets/Pasted%20image%2020241228164305.png)

## Overall

- Got through it in a few hours. Longer than expected. Not a fan of how implicit so much of it is.

