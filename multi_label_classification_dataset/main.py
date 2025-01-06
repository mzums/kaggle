from fastai.vision.all import *

path = Path("/home/mzums/.cache/kagglehub/datasets/meherunnesashraboni/multi-label-image-classification-dataset/versions/7/multilabel_modified/")
images_path = path / "images"

df = pd.read_csv(path / "multilabel_classification_7.csv")
df.columns = [col.strip() for col in df.columns]

def get_x(r): 
    return images_path / r['Image_Name']

def get_y(r): 
    return r['Classes'].split()

dblock = DataBlock(
    blocks=(ImageBlock, MultiCategoryBlock),
    get_x=get_x,
    get_y=get_y,
    item_tfms=Resize(224),
    splitter=RandomSplitter(valid_pct=0.2)
)

dls = dblock.dataloaders(df)

def accuracy_multi(inp, targ, thresh=0.5, sigmoid=True):
    if sigmoid: inp = inp.sigmoid()
    return ((inp>thresh) == targ.bool()).float().mean()

learn = vision_learner(dls, resnet50, metrics = partial(accuracy_multi, thresh=0.2))
learn.fine_tune(3, base_lr=3e-3, freeze_epochs=4)

learn.metrics = partial(accuracy_multi, thresh=0.5)

learn.show_results(ds_idx=1, nrows=3, figsize=(6,8))