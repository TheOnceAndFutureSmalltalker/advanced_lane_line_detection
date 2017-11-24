import numpy as np
import glob
import scipy.misc
import lanelines as ll



# instantiate and initialize processing objects
cc = ll.CameraCalibrator(9, 6, 'camera_cal/calibration*.jpg')
gt = ll.GradientTransformer()
ct = ll.ColorTransformer() 
src = np.float32([[178,720], [573,462], [709,462], [1135,720]])
dst = np.float32([[337,720], [337,0], [970,0], [970,720]])
pw = ll.PerspectiveWarper(src, dst)
llf = ll.LaneLineFinder(10, 40, 25)
pl = ll.Pipeline(cc, gt, ct, pw, llf)      
pl.num_hist = 3           
pl.max_bad_frames = 2  
pl.min_lane_width_ratio = 0.9
pl.x_thresh = (20, 100)   
pl.s_thresh = (150, 255)  


from moviepy.editor import VideoFileClip

def process_video(in_clip_name, out_clip_name):
    """process the video, this must be run command line!"""
    in_clip = VideoFileClip(in_clip_name) #.subclip(0,26) 
    out_clip = in_clip.fl_image(pl.process_image)
    out_clip.write_videofile(out_clip_name, audio=False)
    print('total bad {0}'.format(pl.total_bad))
  
process_video('project_video.mp4', 'project_video_test.mp4') 


#def create_test_frames(in_clip_name, out_clip_name, start, stop):
#    """Create test frames from a second or two of video."""
#    in_clip = VideoFileClip(in_clip_name).subclip(start, stop) 
#    out_clip = in_clip.fl_image(pl.save_image)
#    out_clip.write_videofile(out_clip_name, audio=False)
#    
#create_test_frames('project_video.mp4', 'project_video_test.mp4', 20, 26)  


def process_test_frames():
    """process test frames (about 50 or 2 seconds worth) of project video 
       that have been previously saved out to a folder.
    """
    filenames = glob.glob('pipeline_test/test*.jpg')
    for filename in filenames:
        img = scipy.misc.imread(filename)
        pl.process_image(img)
    print('total bad {0}'.format(pl.total_bad))

#process_test_frames()

  


    

    
