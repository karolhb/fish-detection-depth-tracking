import os
import pandas as pd
import glob
import shutil

def list_all_files(img_folder, out_filename):
    """
    :params:
        img_folder = folder with images to be listed
        out_filename = name of text file with listed images
    """
    img_rows = []

    for file in os.listdir(img_folder):
        if file.endswith(".jpg"):
            img_rows.append('merged_dataset/'+file)
    df = pd.DataFrame(img_rows)
    print(df)

    df.to_csv(out_filename, index=False, header=False)

def rename_files(directory):
    """
    :params:
        directory = directory with files to be renamed
    """
    for count, filename in enumerate(os.listdir(directory)): 
        src = directory + filename

        filetype = '.' + filename.split('.')[-1]
        newname = filename[:-4] + 'orig' + filetype
        dst = directory + newname

        print(newname)

        # rename() function will rename all the files 
        os.rename(src, dst) 

def copy_files_to_new_dir(old_dir, left_name, right_name, left_new_dir, right_new_dir):
    imagesL = glob.glob(old_dir + left_name, recursive=False)
    imagesR = glob.glob(old_dir + right_name, recursive=False)

    # remove all non-pairs
    # go through images in left and right dir, and check if there is a match for all files
    for left_path in imagesL:
        cut_path = left_path.split('.rf')[0]
        right_cut_path = cut_path.replace('out1', 'out2')

        if not any(right_cut_path in r for r in imagesR):
            print("no matches for", right_cut_path)
            nomatch = [s for s in imagesL if cut_path in s]
            for n in nomatch:
                imagesL.remove(n)
    
    for right_path in imagesR:
        cut_path = right_path.split('.rf')[0]
        left_cut_path = cut_path.replace('out2', 'out1')

        if not any(left_cut_path in l for l in imagesL):
            print("no matches for", left_cut_path)
            nomatch = [s for s in imagesR if cut_path in s]
            for n in nomatch:
                imagesR.remove(n)

    for file in imagesL:
        shutil.copy(file,left_new_dir)

    for file in imagesR:
        shutil.copy(file,right_new_dir)


if __name__ == "__main__":
    old_dir = r"Y:\Yolo_v4\darknet\build\darknet\x64\FishFinsDataset"
    left_name = '\*out1*.jpg'
    right_name = '\*out2*.jpg'
    left_new_dir = r"Y:\Yolo_v4\darknet\build\darknet\x64\UWStereoNet_disparity\data\fishfins\train\left"
    right_new_dir = r"Y:\Yolo_v4\darknet\build\darknet\x64\UWStereoNet_disparity\data\fishfins\train\right"
    copy_files_to_new_dir(old_dir, left_name, right_name, left_new_dir, right_new_dir)