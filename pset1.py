import streamlit as st
import fastai
import fastcore
from fastai.vision.all import *
from fastai.vision.widgets import *
from fastdownload import download_url
from duckduckgo_search import DDGS
from fastcore.all import *
from PIL import Image
import shutil
import os

st.title("Mongolian Foods Classifier")

# Define the function to search for images
def search_images(term, max_images=30):
    st.write(f"Searching for '{term}'")
    with DDGS() as ddgs:
        search_results = ddgs.images(keywords=term, safesearch='moderate')
        image_urls = [result.get("image") for result in search_results if result.get("image")]
        return L(image_urls[:max_images])

# Define paths and food types
food_types = 'buuz','khuushuur','tsuivan','bansh'
path = Path('mongolian_foods')

# Create directories for each food type if not already present
if not path.exists():
    path.mkdir()
    for food in food_types:
        dest = (path/food)
        dest.mkdir(exist_ok=True)
        urls = search_images(f'{food} Mongolian dumpling', max_images=50)
        download_images(dest, urls=urls)

# Verify and clean up images
fns = get_image_files(path)
failed = verify_images(fns)
if len(failed) > 0:
    st.write(f"Failed images: {len(failed)}")
    failed.map(Path.unlink)

# Remove images that do not match the category manually
for food in food_types:
    food_path = path/food
    for img in get_image_files(food_path):
        try:
            with Image.open(img) as im:
                im.verify()
        except:
            img.unlink()

# Define DataBlock and DataLoaders
foods = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=RandomResizedCrop(224, min_scale=0.5),
    batch_tfms=aug_transforms()
)

dls = foods.dataloaders(path)

# Display a batch of images
st.write("Here are some sample images:")
dls.show_batch(max_n=8, nrows=2)

# Train the model
st.write("Training the model...")
learn = cnn_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(6)

# Show confusion matrix and top losses
interp = ClassificationInterpretation.from_learner(learn)
st.write("Confusion Matrix:")
st.pyplot(interp.plot_confusion_matrix())

st.write("Top losses:")
st.pyplot(interp.plot_top_losses(5, nrows=1))

# Allow user to clean up misclassified images
cleaner = ImageClassifierCleaner(learn)
st.write("Cleaning up misclassified images...")
if st.button('Clean Images'):
    for idx in cleaner.delete():
        cleaner.fns[idx].unlink()
    for idx, cat in cleaner.change():
        shutil.move(str(cleaner.fns[idx]), path/cat)

# Re-load the data after cleaning
dls = foods.dataloaders(path, bs=32)
learn = cnn_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(4)

# Final results
interp = ClassificationInterpretation.from_learner(learn)
st.write("Final Confusion Matrix:")
st.pyplot(interp.plot_confusion_matrix())
st.write("Final Top losses:")
st.pyplot(interp.plot_top_losses(15, nrows=3))

# Save the final trained model
learn.export('mongolian_foods_classifier.pkl')

st.write("Model saved as 'mongolian_foods_classifier.pkl'")
