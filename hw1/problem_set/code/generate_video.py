import glob
import numpy as np
from PIL import Image

def generate_gif(path):
    """
    Generate a gif of stitched frames for submission
    param[in] path : File Path Save Location
    """
    frames = [Image.open(image) for image in glob.glob(f"{path}/*.png")]
    print(np.size(frames))
    frame_one = frames[0]
    frame_one.save("result111.gif", format="GIF", append_images=frames,
               save_all=True, duration=100, loop=0)
    
if __name__ == "__main__":
    generate_gif("./results")