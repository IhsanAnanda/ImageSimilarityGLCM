import argparse
import numpy as np
import cv2 as cv
import pandas as pd 
from skimage.feature import greycomatrix, greycoprops

properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']
sudut = ['0', '45', '90','135']

#GLCM extraction greycomatrix() & greycoprops() for angle 0, 45, 90, 135
def calc_glcm_all_agls(img, props, dists=[5], agls=[0, np.pi/4, np.pi/2, 3*np.pi/4], lvl=256, sym=True, norm=True):
    
    glcm = greycomatrix(img, 
                        distances=dists, 
                        angles=agls, 
                        levels=lvl,
                        symmetric=sym, 
                        normed=norm)
    feature = []
    glcm_props = [propery for name in props for propery in greycoprops(glcm, name)[0]]
    for item in glcm_props:
            feature.append(item)
    
    return feature

#Build GLCM DataFrame
def GLCM_df(Image):
    columns = []
    glcm_all_agls = calc_glcm_all_agls(Image, props=properties)
    for name in properties :
        for ang in sudut:
            columns.append(name + "_" + ang + " ")
    #dataframe GLCM
    glcm_df = pd.DataFrame( glcm_all_agls, columns)
    average = np.average(glcm_all_agls, axis= 0)
    print(glcm_df)
    
    return average


if __name__ == '__main__':
    #Initialzie the parser
    parser = argparse.ArgumentParser(
        description="Image Similarity Using GLCM"
        )
    
    #Parameter
    parser.add_argument('-Img1','--Image1', help="Image 1")
    parser.add_argument('-Img2','--Image2', help="Image 2")
    parser.add_argument('-T','--Threshold', help="Similarity Threshold", type=float, default=1)
    
    #Parse the argument
    args = parser.parse_args()
    print(args)
    
    #Load Image
    gambar1 = cv.imread(args.Image1, 0)
    Image1 = np.array(gambar1)
    print("Image 1 Size : ", Image1.size)
    gambar2 = cv.imread(args.Image2, 0)
    Image2 = np.array(gambar2)
    print("Image 2 Size : ", Image2.size)
    
    #Extract Images GLCM
    print("\nGLCM Image 1 :")
    avg1 = GLCM_df(Image1)
    print("Image 1 Avg GLCM : ", avg1)
    print("\nGLCM Image 2 :")
    avg2 = GLCM_df(Image2)
    print("Image 2 Avg GLCM : ", avg2)
    
    #Checking similarity
    difference = abs(avg1-avg2)
    print(difference)
    if difference>args.Threshold:
        print("\nKedua Gambar Berbeda")
    else:
        print("\nKedua Gambar Sama")
    
    
    
    
    