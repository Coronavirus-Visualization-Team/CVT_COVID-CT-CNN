import os, shutil
from pathlib import Path
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import matplotlib.pyplot as plt

train_batch_size = 10
val_batch_size = 10
image_size = (150, 150)

data_dir = '/Users/vishakh/Documents/12th_Grade/CVT-CGA/CVT_COVID-CT-CNN'

train_dir = os.path.join(data_dir, 'train')
Path(train_dir).mkdir(parents=True, exist_ok=True)

validation_dir = os.path.join(data_dir, 'validation')
Path(validation_dir).mkdir(parents=True, exist_ok=True)

test_dir = os.path.join(data_dir, 'test')
Path(test_dir).mkdir(parents=True, exist_ok=True)

train_txt = os.path.join(data_dir, 'TrainingTXT')
validation_txt = os.path.join(data_dir, 'ValidationTXT')
test_txt = os.path.join(data_dir, 'TestingTXT')

def read(datadir, fname):
    with open(os.path.join(datadir, fname), 'r') as f:
        data_fnames = [line.rstrip('\n') for line in f.readlines()]
    return data_fnames

def createData(dirname, arr, original_data_dir):
    Path(dirname).mkdir(parents=True, exist_ok=True)
    for fname in arr:
        src = os.path.join(data_dir, original_data_dir, fname)
        dst = os.path.join(dirname, fname)
        shutil.copyfile(src, dst)

def initData():
    training_covid = read(train_txt, 'trainCT_COVID.txt')
    createData(os.path.join(train_dir, 'COVID'), training_covid, 'CT_COVID')
    training_non_covid = read(train_txt, 'trainCT_NonCOVID.txt')
    createData(os.path.join(train_dir, 'NonCOVID'), training_non_covid, 'CT_NonCOVID')
    num_train_samples = len(training_covid) + len(training_non_covid)

    validation_covid = read(validation_txt, 'valCT_COVID.txt')
    createData(os.path.join(validation_dir, 'COVID'), validation_covid, 'CT_COVID')
    validation_non_covid = read(validation_txt, 'valCT_NonCOVID.txt')
    createData(os.path.join(validation_dir, 'NonCOVID'), validation_non_covid, 'CT_NonCOVID')
    num_val_samples = len(validation_covid) + len(validation_non_covid)

    test_covid = read(test_txt, 'testCT_COVID.txt')
    createData(os.path.join(test_dir, 'COVID'), test_covid, 'CT_COVID')
    test_non_covid = read(test_txt, 'testCT_NonCOVID.txt')
    createData(os.path.join(test_dir, 'NonCOVID'), test_non_covid, 'CT_NonCOVID')

    return (num_train_samples, num_val_samples)

def preprocess_data():
    train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode="nearest"
                )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=train_batch_size,
        class_mode='binary'
    )

    validation_generator = train_datagen.flow_from_directory(
        validation_dir,
        target_size=image_size,
        batch_size=val_batch_size,
        class_mode='binary'
    )

    return (train_generator, validation_generator)

def test_augmentation():
    datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode="nearest"
            )

    fnames = [os.path.join(train_dir, 'COVID', fname) for fname in os.listdir(os.path.join(train_dir, 'COVID'))]
    img_path = fnames[5]

    img = image.load_img(img_path, target_size=image_size)
    x = image.img_to_array(img)
    x = x.reshape((1,) + x.shape)
    print(x.shape)
    i = 0
    for batch in datagen.flow(x, batch_size=1):
        plt.figure(i)
        imgplot = plt.imshow(image.array_to_img(batch[0]))
        i += 1
        if (i % 4 == 0):
            break
    plt.show()
