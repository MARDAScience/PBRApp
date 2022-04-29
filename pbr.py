import streamlit as st
from PIL import Image
import os, io, glob
import zipfile

# import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2hsv, hsv2rgb
from skimage.restoration import denoise_wavelet, estimate_sigma, rolling_ball

from functools import partial
# rescale_sigma=True required to silence deprecation warnings
_denoise_wavelet = partial(denoise_wavelet, rescale_sigma=True)
from skimage import util

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=RuntimeWarning)

# =========================================================
def rescale(dat,mn,mx):
    """
    rescales an input dat between mn and mx
    """
    m = np.nanmin(dat.flatten())
    M = np.nanmax(dat.flatten())
    return (mx-mn)*(dat-m)/(M-m)+mn

# =========================================================
def sharpen(Z, radius):

    sigma_est = estimate_sigma(Z, multichannel=True, average_sigmas=False)
    region = denoise_wavelet(Z, multichannel=True, rescale_sigma=True, wavelet_levels=6, convert2ycbcr=True,
                              method='BayesShrink', mode='soft', sigma=np.max(sigma_est)*5)
    original = rescale(region,0,255)

    Zo = np.ma.filled(original, fill_value=np.nan).copy()
    hsv = rgb2hsv(Zo)
    im = (0.299 * Zo[:,:,0] + 0.5870*Zo[:,:,1] + 0.114*Zo[:,:,2])
    im[Z[:,:,0]==0]=0

    ##https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_rolling_ball.html#sphx-glr-auto-examples-segmentation-plot-rolling-ball-py
    ##background = rolling_ball(im, radius=100)
    image_inverted = util.invert(im)
    background_inverted = rolling_ball(image_inverted, radius=radius)
    filtered_image_inverted = image_inverted - background_inverted
    filtered_image = util.invert(filtered_image_inverted)
    background = util.invert(background_inverted)

    background[np.isnan(background)] = 0
    background[np.isinf(background)] = 0
    intensity = (im/background)
    intensity[np.isnan(intensity)] = 0
    intensity[np.isinf(intensity)] = 0
    intensity = (255*intensity).astype('uint8')

    sharpened = hsv2rgb(np.dstack([hsv[:,:,0], hsv[:,:,1], intensity]))

    sharpened[:,:,0] = rescale(sharpened[:,:,0],Z[:,:,0].min(), Z[:,:,0].max())
    sharpened[:,:,1] = rescale(sharpened[:,:,1],Z[:,:,1].min(), Z[:,:,2].max())
    sharpened[:,:,2] = rescale(sharpened[:,:,2],Z[:,:,2].min(), Z[:,:,1].max())
    sharpened = (sharpened).astype('uint8')

    return sharpened.astype('uint8')


#============================================================
# =========================================================

def do_filter(f, outfile,radius=5):
    #Z = imread(f)
    Z = np.array(Image.open(f), dtype=np.uint8)
    #radius = 3
    sharpened = sharpen(Z, radius)
    #imwrite(outfile,sharpened)
    im = Image.fromarray(sharpened)
    im.save(outfile, "JPEG")

def rm_thumbnails():
    try:
        for k in glob.glob('*filt*'):
            os.remove(k)
    except:
        pass

st.set_page_config(
     page_title="Create pbr images",
     page_icon="",
     layout="centered",
     initial_sidebar_state="collapsed",
     menu_items={
         'Get Help': None,
         'Report a bug': None,
         'About': "PBR your images!"
     }
 )

uploaded_files = st.file_uploader("Upload files. Works with jpg files only", accept_multiple_files=True)
for uploaded_file in uploaded_files:
     bytes_data = uploaded_file.read()
images_list=uploaded_files

# Initialize Sniffer's states
if 'img_idx' not in st.session_state:
    st.session_state.img_idx=0

def create_zip():
    with zipfile.ZipFile('resized_images.zip', mode="w") as archive:
        for k in glob.glob("*filt*"):
            archive.write(k)
    
    with open('resized_images.zip','rb') as f:
        g=io.BytesIO(f.read()) 
    os.remove('resized_images.zip')
    rm_thumbnails()
    return g

# def do_resize(infile, outfile):
#     size = 512, 512
#     im = Image.open(infile)
#     im.thumbnail(size, Image.ANTIALIAS)
#     im.save(outfile, "JPEG")

def filter_button():
    for k in range(len(images_list)):
        infile = images_list[k]
        #outfile = os.path.splitext(infile.name)[0] + ".thumbnail"
        outfile = infile.name.replace('.','_filt.')
        do_filter(io.BytesIO(infile.getvalue()), outfile)
        #do_resize(io.BytesIO(infile.getvalue()), outfile)
        print(outfile)        
        st.session_state.img_idx += 1

# def next_button():
#     if -1 < st.session_state.img_idx < len(images_list)-1:
#         row={"Filename":images_list[st.session_state.img_idx],'Sorted':"bad",'Index':st.session_state.img_idx}
#         st.session_state.img_idx += 1
#     else:
#         st.warning('No more images')

if images_list==[]:
    image= Image.open("./assets/IMG_0202_filt.JPG")
else:
    if st.session_state.img_idx>=len(images_list):
        image = Image.open("./assets/IMG_0202_filt.JPG")
    else:
        image = Image.open(images_list[st.session_state.img_idx])

st.title("P.B.R. Filter.")
st.markdown("by Daniel Buscombe, Marda Science. {P}ansharpening by {B}ackground {R}emoval algorithm for sharpening RGB images. Upload images. Download a zipped folder of filtered images. See [github page](https://github.com/dbuscombe-usgs/PBR_filter) for code and docs. App based on [Sniffer streamlit app](https://github.com/2320sharon/1_streamlit) by the fantastic [@2320sharon](https://github.com/2320sharon)")
st.image("./assets/IMG_0202_filt.JPG")
# Sets num_image=1 if images_list is empty
num_images=(len(images_list)) if (len(images_list))>0 else 1
my_bar = st.progress((st.session_state.img_idx)/num_images)

col1,col2,col3,col4=st.columns(4)
with col1:
    st.button(label="Filter",key="filter_button",on_click=filter_button)
    # st.button(label="View next image",key="next_button",on_click=no_button)    
with col2:
    # Display done.jpg when all images are sorted 
    if st.session_state.img_idx>=len(images_list):
        image = Image.open("./assets/IMG_0202_filt.JPG")
        st.image(image,width=300)
    else:
        # caption is none when images_list is empty otherwise it is the image name 
        caption = '' if images_list==[] else f'#{st.session_state.img_idx} {images_list[st.session_state.img_idx].name}'
        st.image(image, caption=caption,width=300)
    
with col4:
    st.download_button(
     label="Download filtered imagery",
     data=create_zip(),
     file_name= 'filtered_images.zip', 
 )

#