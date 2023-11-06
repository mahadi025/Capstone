import cv2
import os
import shutil
import splitfolders
import Augmentor


def median_filter(img, ksize=7):
    output = cv2.medianBlur(img, ksize)
    return output

def gaussian_filter(img):
    output = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)
    return output

def total_images(data_dir):
    total_images = 0
    for image_class in os.listdir(data_dir):
        print(image_class+':'+str(len(os.listdir(data_dir+'/'+image_class))))
        total_images += len(os.listdir(data_dir+'/'+image_class))
    print(f'Total Images:{total_images}')
    return total_images

def split_dataset(loc, data_dir='dataset',split=2):

    if os.path.isdir(f'{data_dir}/split'):
        shutil.rmtree(f'{data_dir}/split')
    if split==3:
        splitfolders.ratio(loc, output=f'{data_dir}/split',ratio=(0.70, 0.20, 0.10))
    else:
        splitfolders.ratio(loc, output=f'{data_dir}/split',ratio=(0.90, 0.10))
        os.rename(f'{data_dir}/split/val', f'{data_dir}/split/test')


def resize(input_images_dir, image_size=(224,224), selected_disease=None):
    resized_images_path='dataset/resized_images'
    if os.path.isdir(resized_images_path):
        shutil.rmtree(resized_images_path)
    diseases_name = []
    if selected_disease is not None:
        for image_class in (selected_disease):
            diseases_name.append(image_class)
    else:
        for image_class in os.listdir(input_images_dir):
            diseases_name.append(image_class)
    if not os.path.isdir(resized_images_path):
        for disease in diseases_name:
            os.makedirs(f'{resized_images_path}/{disease}')
    for image_class in os.listdir(resized_images_path):
        for image in os.listdir(os.path.join(input_images_dir, image_class)):
            image_path = os.path.join(input_images_dir, image_class, image)
            try:
                img = cv2.imread(image_path)
                img=cv2.resize(img, image_size)
                cv2.imwrite(os.path.join(resized_images_path+f'/{image_class}', image), img)
            except Exception as e:
                print(e)
    print(f'Resized images are saved in: {resized_images_path}')

def contrast_adjustment(input_images_dir, alpha=1.3, beta=1, selected_disease=None):
    contrast_adjustment_path='dataset/contrast_adjustment_images'
    if os.path.isdir(contrast_adjustment_path):
        shutil.rmtree(contrast_adjustment_path)
    diseases_name = []
    if selected_disease is not None:
        for image_class in (selected_disease):
            diseases_name.append(image_class)
    else:
        for image_class in os.listdir(input_images_dir):
            diseases_name.append(image_class)
    if not os.path.isdir(contrast_adjustment_path):
        for disease in diseases_name:
            os.makedirs(f'{contrast_adjustment_path}/{disease}')
    for image_class in os.listdir(input_images_dir):
        for image in os.listdir(os.path.join(input_images_dir, image_class)):
            image_path = os.path.join(input_images_dir, image_class, image)
            try:
                img = cv2.imread(image_path)
                img=cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
                cv2.imwrite(os.path.join(contrast_adjustment_path+f'/{image_class}', image), img)
            except Exception as e:
                print(e)
    print(f'Contrast adjusted images are saved in: {contrast_adjustment_path}')


def smoothening(input_images_dir, selected_disease=None):
    smoothen_images_path='dataset/smoothen_images'
    if os.path.isdir(smoothen_images_path):
        shutil.rmtree(smoothen_images_path)
    diseases_name = []
    if selected_disease is not None:
        for image_class in (selected_disease):
            diseases_name.append(image_class)
    else:
        for image_class in os.listdir(input_images_dir):
            diseases_name.append(image_class)

    if not os.path.isdir(smoothen_images_path):
        for disease in diseases_name:
            os.makedirs(f'{smoothen_images_path}/{disease}')
    for image_class in os.listdir(input_images_dir):
        for image in os.listdir(os.path.join(input_images_dir, image_class)):
            image_path = os.path.join(input_images_dir, image_class, image)
            try:
                img = cv2.imread(image_path)
                img=median_filter(img)
                img=gaussian_filter(img)
                cv2.imwrite(os.path.join(smoothen_images_path+f'/{image_class}', image), img)
            except Exception as e:
                print(e)
    print(f'Smoothen images are saved in: {smoothen_images_path}')

def grayscale(input_images_dir, selected_disease=None):
    gray_images_path='dataset/gray_images'
    if os.path.isdir(gray_images_path):
        shutil.rmtree(gray_images_path)
    diseases_name = []
    if selected_disease is not None:
        for image_class in (selected_disease):
            diseases_name.append(image_class)
    else:
        for image_class in os.listdir(input_images_dir):
            diseases_name.append(image_class)
    if not os.path.isdir(gray_images_path):
        for disease in diseases_name:
            os.makedirs(f'{gray_images_path}/{disease}')
    for image_class in os.listdir(input_images_dir):
        for image in os.listdir(os.path.join(input_images_dir, image_class)):
            image_path = os.path.join(input_images_dir, image_class, image)
            try:
                img = cv2.imread(image_path, 0)
                cv2.imwrite(os.path.join(gray_images_path+f'/{image_class}', image), img)
            except Exception as e:
                print(e)
    print(f'Gray images are saved in: {gray_images_path}')


def rename(input_images_dir, selected_disease=None):
    renamed_images_path='dataset/renamed_images'
    if os.path.isdir(renamed_images_path):
        shutil.rmtree(renamed_images_path)
    diseases_name = []
    if selected_disease is not None:
        for image_class in (selected_disease):
            diseases_name.append(image_class)
    else:
        for image_class in os.listdir(input_images_dir):
            diseases_name.append(image_class)
    if not os.path.isdir(renamed_images_path):
        for disease in diseases_name:
            os.makedirs(f'{renamed_images_path}/{disease}')
    image_numbers={}
    for disease in diseases_name:
        image_numbers[disease]=1
    for image_class in os.listdir(input_images_dir):
        for image in os.listdir(os.path.join(input_images_dir, image_class)):
            image_path = os.path.join(input_images_dir, image_class, image)
            try:
                img = cv2.imread(image_path)
                cv2.imwrite(f'{renamed_images_path}/{image_class}/{image_class}_{(image_numbers[image_class])}.jpg',img)
                image_numbers[image_class]+=1
            except Exception as e:
                print(e)
    print(f'Renamed images are saved in: {renamed_images_path}')


def augmentation(input_images_dir, times=10):
    if os.path.isdir('dataset/augmented_images'):
        shutil.rmtree('dataset/augmented_images')
    p = Augmentor.Pipeline(input_images_dir, output_directory='../augmented_images')
    p.rotate_random_90(probability=1)
    p.flip_left_right(probability=1)
    p.flip_top_bottom(probability=1)
    p.rotate180(probability=1)

    p.rotate_random_90(probability=1)
    p.flip_top_bottom(probability=1)
    p.flip_left_right(probability=1)
    p.rotate180(probability=1)

    num_augmented_images = total_images(input_images_dir)*times
    p.sample(num_augmented_images)
    print(f'Augmented images are saved in: dataset/augmented_images')