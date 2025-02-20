#####################################################################

# Example : real-time mosaicking - skeleton outline of functionality

# takes input from a video file specified on the command line
# (e.g. python FILE.py video_file) or from an attached web camera

# Base File Author : Toby Breckon, toby.breckon@durham.ac.uk

# Student Author(s) : <INSERT NAMES>

# Copyright (c) <YEAR> <INSERT NAMES>, Durham University, UK
# Copyright (c) 2017-21 Toby Breckon, Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

#####################################################################

import cv2
import sys
import argparse

#####################################################################

# import all the provided helper functions

import mosaic_support as ms

#####################################################################

# check OpenCV version and if extra modules are present

print("\nOpenCV: " + cv2.__version__)
print("OpenCV Extra Modules Present: " +
      str(ms.extra_opencv_modules_present()))
print("OpenCV Non-Free Algorithms Present: " +
      str(ms.non_free_algorithms_present()))
print("Python: " + sys.version)
print()

#####################################################################

keep_processing = True

# parse command line arguments for camera ID or video file

parser = argparse.ArgumentParser(
    description='Perform ' +
    sys.argv[0] +
    ' operation on incoming camera/video image')
parser.add_argument(
    "-c",
    "--camera_to_use",
    type=int,
    help="specify camera to use",
    default=0)
parser.add_argument(
    "-r",
    "--rescale",
    type=float,
    help="rescale image by this factor",
    default=1.0)
parser.add_argument(
    'video_file',
    metavar='video_file',
    type=str,
    nargs='?',
    help='specify optional video file')
args = parser.parse_args()

#####################################################################

# define video capture object

try:
    # to use a non-buffered camera stream (via a separate thread)

    if not (args.video_file):
        import camera_stream
        cap = camera_stream.CameraVideoStream()
    else:
        cap = cv2.VideoCapture()  # not needed for video files

except BaseException:
    # if not then just use OpenCV default

    print("INFO: camera_stream class not found - camera input may be buffered")
    cap = cv2.VideoCapture()

# define display window names

window_name_live = "Live Camera Input"  # window name
window_name_mosaic = "Mosaic Output"

# initially set our mosaic to an empty image

mosaic = None

# if command line arguments are provided try to read video_file
# otherwise default to capture from attached camera

if (((args.video_file) and (cap.open(str(args.video_file))))
        or (cap.open(args.camera_to_use))):

    # create windows by name (as resizable)

    cv2.namedWindow(window_name_live, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_name_mosaic, cv2.WINDOW_NORMAL)

    while (keep_processing):

        # if video file successfully open then read frame from video

        if (cap.isOpened):
            ret, frame = cap.read()

            # when we reach the end of the video (file) exit cleanly

            if (ret == 0):
                keep_processing = False
                continue

            # rescale if specified

            if (args.rescale != 1.0):
                frame = cv2.resize(
                    frame, (0, 0), fx=args.rescale, fy=args.rescale)

        # *** BEGIN TODO - outline of required mosaicking code ***

        # detect features in current image
        im_features, im_descriptors = ms.get_features(frame, 400) # 400-1000 surf thresh.
        n_features = len(im_features)

        # if enough features present in image
        if n_features > 500:

            # if current mosaic image is empty (i.e. at start of process)
            if mosaic is None:

                # copy current frame to mosaic image
                mosaic = frame

                # continue to next frame (i.e. next loop iteration)
                continue

            # else
            else:

                # get features in current mosaic (or similar)
                # (may need to check features are found, or can assume OK)
                mos_features, mos_descriptors = ms.get_features(mosaic, 400)


                # compute matches between current image and mosaic
                # (cv2.drawMatches() may be useful for debugging here)
                n_checks = 50
                match_ratio = 0.7
                matches = ms.match_features(im_descriptors, mos_descriptors, n_checks, match_ratio)
                cv2.drawMatches(frame, im_features, mosaic, mos_features, matches, frame)
                print(len(matches))

                # compute homography H between current image and mosaic
                H, mask = ms.compute_homography(im_features, mos_features, matches)

                # calculate the required size of the new mosaic image
                # if we add the current frame into it
                size, offset = ms.calculate_size(frame.shape, mosaic.shape, H)
                print(offset)

                # if offset[0] + offset[1] < 10:
                    # print('Offset too small')
                    # continue # skip if offset is too small

                # merge the current frame into the new mosaic using
                # knowldge of homography H + required size of image
                mosaic = ms.merge_images(frame, mosaic, H, size, offset)

                # (optional) - resize output mosaic to be % of full size
                # so it fits on screen or scale in porportion to screen size

        # else when not enough features present in image
        else:
            # (cv2.drawKeypoints() may be useful for debugging here)
            # cv2.drawKeypoints(frame, im_features, frame, color=(0, 255, 0))

            # continue to next frame (i.e. next loop iteration)
            continue

        # if (mosaic is None):  # *** TODO REMOVE this part ***
        #     mosaic = frame  # only here so code runs at first time

        # *** END TODO outline of required mosaicking code ***

        # display input and output (perhaps consider use of
        # cv2.WND_PROP_FULLSCREEN)

        cv2.imshow(window_name_live, frame)
        cv2.imshow(window_name_mosaic, mosaic)

        # start the event loop - wait 500ms (i.e. 1000ms / 2 fps = 500 ms)

        key = cv2.waitKey(500) & 0xFF

        # detect specific key strokes
        # e.g. if user presses "x" then exit

        if (key == ord('x')):
            keep_processing = False

    # close all windows

    cv2.destroyAllWindows()

else:
    print("No video file specified or camera connected.")

#####################################################################
