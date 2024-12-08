import os
from datetime import datetime, timedelta
import requests
import cv2
from cv2 import dnn_superres
import numpy as np

# Base URL for WCS
WCS_BASE_URL = "https://view.eumetsat.int/geoserver/wcs"

# Directory to save images
output_dir = "eumetsat_images"
upscaled_dir = os.path.join(output_dir, "upscaled")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(upscaled_dir, exist_ok=True)

# Function to fetch a specific coverage
def get_coverage(coverage_id, output_file, bbox, time_range, crs="EPSG:4326", format="image/png"):
    """
    Fetches a specific coverage from the WCS with a given time range.
    """
    CQL_FILTER = f"bbox(the_geom, {str(bbox)[1:-1]}) AND time DURING {time_range}"

    params = {
        "service": "WCS",
        "version": "2.0.1",
        "request": "GetCoverage",
        "coverageId": coverage_id,
        "format": format,
        "CQL_FILTER": CQL_FILTER,
        "subset": [f"Lat({bbox[1]},{bbox[3]})", f"Long({bbox[0]},{bbox[2]})"],
        "crs": crs,
    }

    response = requests.get(WCS_BASE_URL, params=params, stream=True)
    if response.status_code == 200:
        with open(output_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return output_file
    else:
        print(f"Failed to fetch coverage: {response.status_code}")
        print(response.text)
        return None

# Function to download images at regular intervals
def download_images(coverage_id, bbox, start_time, end_time, interval_minutes=10):
    """
    Downloads satellite images at regular intervals within a given time range.
    """
    images = []
    current_time = datetime.fromisoformat(start_time)
    end_time = datetime.fromisoformat(end_time)

    while current_time <= end_time:
        timestamp = current_time.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        next_timestamp = (current_time + timedelta(minutes=interval_minutes)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        time_range = f"{timestamp}/{next_timestamp}"
        output_file = os.path.join(output_dir, f"{coverage_id}_{timestamp}.png")
        
        if os.path.exists(output_file):
            print(f"File already exists, skipping download: {output_file}")
            images.append(output_file)
        else:
            print(f"Downloading image for time range: {time_range}")
            downloaded_file = get_coverage(coverage_id, output_file, bbox, time_range)
            if downloaded_file:
                images.append(downloaded_file)
        
        current_time += timedelta(minutes=interval_minutes)

    return images

# Function to upscale images using EDSR
def upscale_images(images, model_path, scale=4):
    """
    Upscales a list of images using the EDSR model, skipping already upscaled ones.
    """
    sr = dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel("lapsrn", scale)

    upscaled_images = []

    for img_path in images:
        # Determine the corresponding upscaled image path
        upscaled_path = os.path.join(upscaled_dir, os.path.basename(img_path))

        if os.path.exists(upscaled_path):
            print(f"Upscaled image already exists, skipping: {upscaled_path}")
            upscaled_images.append(upscaled_path)
        else:
            print(f"Upscaling image: {img_path}")
            image = cv2.imread(img_path)
            if image is not None:
                upscaled_image = sr.upsample(image)
                cv2.imwrite(upscaled_path, upscaled_image)
                upscaled_images.append(upscaled_path)
                print(f"Saved upscaled image to {upscaled_path}")
            else:
                print(f"Failed to read image: {img_path}")

    return upscaled_images

# Function to interpolate frames using optical flow
def interpolate_frames_optical_flow(images, fps=30, interp_factor=1,upscale=4):
    """
    Interpolates between frames using optical flow to create a smooth animation.
    """
    interpolated_frames_fwd = []
    interpolated_frames_bwd = []
    interpolated_frames = []
    
    for i in range(0, len(images)-1):
        img2 = cv2.imread(images[i+1])
        img1 = cv2.imread(images[i])

        # sharpening


        if img2 is not None and img1 is not None:
            # Ensure both frames are the same size
            img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))

            # Convert to grayscale for optical flow
            gray1 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

            scaled_gray1 = cv2.resize(gray1, None, fx=1/upscale, fy=1/upscale, interpolation=cv2.INTER_AREA)
            scaled_gray2 = cv2.resize(gray2, None, fx=1/upscale, fy=1/upscale, interpolation=cv2.INTER_AREA)

            # Compute dense optical flow
            flowfwd = cv2.calcOpticalFlowFarneback(
                scaled_gray1, scaled_gray2, None, 0.5, 5, 30, 1, 5, 1.2, 0
            )

            flowbwd = cv2.calcOpticalFlowFarneback(
                scaled_gray2, scaled_gray1, None, 0.5, 5, 30, 1, 5, 1.2, 0
            )

            flowfwd = cv2.resize(flowfwd, (gray1.shape[1], gray1.shape[0]))
            flowbwd = cv2.resize(flowbwd, (gray2.shape[1], gray2.shape[0]))
            flowfwd *= upscale
            flowbwd *= upscale

            # Interpolate intermediate frames
            interpolated_frames_fwd.append(img1)
            interpolated_frames_bwd.append(img1)
            # interpolated_frames_fwd.append(img1)

            for j in range(1, interp_factor):
                alpha = j / interp_factor
                alpha_rev = 1 - alpha

                map_x_fwd, map_y_fwd = np.meshgrid(
                    np.arange(img1.shape[1]), np.arange(img1.shape[0])
                )
                map_x_fwd = (map_x_fwd + (flowfwd[..., 0]) * alpha).astype(np.float32)
                map_y_fwd = (map_y_fwd + (flowfwd[..., 1]) * alpha).astype(np.float32)

                map_x_bwd, map_y_bwd = np.meshgrid(
                    np.arange(img2.shape[1]), np.arange(img2.shape[0])
                )

                map_x_bwd = (map_x_bwd + (flowbwd[..., 0]) * alpha_rev).astype(np.float32)
                map_y_bwd = (map_y_bwd + (flowbwd[..., 1]) * alpha_rev).astype(np.float32)

                intermediate_frame_rev = cv2.remap(
                    img2, map_x_bwd, map_y_bwd, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
                )
                interpolated_frames_bwd.append(intermediate_frame_rev)

                # Warp the image using the flow field
                intermediate_frame_fwd = cv2.remap(
                    img1, map_x_fwd, map_y_fwd, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
                )
                interpolated_frames_fwd.append(intermediate_frame_fwd)

                intermediate_interpolated_frames = cv2.addWeighted(
                    intermediate_frame_fwd, alpha_rev, intermediate_frame_rev, alpha, 0
                )

            #    #last frame
            #     # lastframe = interpolated_frames
            #     # schlieren = cv2.subtract(lastframe, intermediate_interpolated_frames)

            #     schlieren = cv2.subtract(interpolated_frames, intermediate_interpolated_frames)
            #     # print(schlieren)

            #     min_val = schlieren.min()
            #     max_val = schlieren.max()
            #     # schlieren_normalized = cv2.normalize(schlieren, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            #     # Convert to a colormap (red for positive, blue for negative)
            #     schlieren_colormap = cv2.applyColorMap(schlieren, cv2.COLORMAP_JET)



            #     schlieren = cv2.addWeighted(intermediate_interpolated_frames, 1, schlieren_colormap, 1, 0)


                interpolated_frames.append(intermediate_interpolated_frames)

            


                
            
            interpolated_frames_fwd.append(img2)
            interpolated_frames_bwd.append(img2)
            interpolated_frames.append(img2)


    return interpolated_frames

def frames_to_video(frames, output_video, fps=30):
    """
    Converts a list of frames to a video.
    """
    height, width, layers = frames[0].shape
    size = (width, height) 
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for frame in frames:
        out.write(frame)

    out.release()

    print(f"Video saved to: {output_video}")

def schlieren2(images,overlay_alpha=0.2,scale = 1):

    schlieren_frames = []
    schlieren = []
    
    for i in range(0, len(images)-1):
        img2 = (images[i+1])
        img1 = (images[i])

        # sharpening


        if img2 is not None and img1 is not None:
            # Ensure both frames are the same size
            img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))

            # Convert to grayscale for optical flow
            gray1 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

            scaled_gray1 = cv2.resize(gray1, None, fx=1/scale, fy=1/scale, interpolation=cv2.INTER_AREA)
            scaled_gray2 = cv2.resize(gray2, None, fx=1/scale, fy=1/scale, interpolation=cv2.INTER_AREA)

            # Compute dense optical flow
            flowfwd = cv2.calcOpticalFlowFarneback(
                scaled_gray1, scaled_gray2, None, 0.5, 5, 30, 1, 5, 1.2, 0
            )

            flowfwd = cv2.resize(flowfwd, (gray1.shape[1], gray1.shape[0]))
            flowfwd *= scale


            alpha = 1

            map_x_fwd, map_y_fwd = np.meshgrid(
                np.arange(img1.shape[1]), np.arange(img1.shape[0])
            )
            map_x_fwd = (map_x_fwd + (flowfwd[..., 0]) * alpha).astype(np.float32)
            map_y_fwd = (map_y_fwd + (flowfwd[..., 1]) * alpha).astype(np.float32)

            # Warp the image using the flow field
            intermediate_frame_fwd = cv2.remap(
                img1, map_x_fwd, map_y_fwd, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
            )

            img1 = intermediate_frame_fwd
    
            img1_int = img1.astype(np.int16)
            img1_int -= 128

            img2_int = img2.astype(np.int16)
            img2_int -= 128

            scale_sch = 40

            gray_diff = cv2.subtract(img2_int, img1_int)*scale_sch
            gray_diff = np.clip(gray_diff, -127, 127)
            gray_diff[(gray_diff > -(2*scale_sch)) & (gray_diff < (2*scale_sch))] = 0



            grey_diff_uint8 = (gray_diff+128).astype(np.uint8)
            gray_diff_g = cv2.cvtColor(grey_diff_uint8, cv2.COLOR_BGR2GRAY)
            grey_diff_g = np.clip(gray_diff_g, 0, 255).astype(np.uint8)

            # Apply a colormap for visualization
            schlieren_img = cv2.applyColorMap(gray_diff_g, cv2.COLORMAP_JET)

            # Resize Schlieren image to match the original dimensions (if needed)
            schlieren_img = cv2.resize(schlieren_img, (img2.shape[1], img2.shape[0]))

            # add depth to the image set to 1 for now

            # find the mode colour
            # mode = np.argmax(np.bincount(schlieren_img.flatten()))

            # get rid of the mode colour setting it to black 
            # schlieren_img[schlieren_img == mode] 

            # if colour = 126 255 130 set it to 0 0 0 
            mask = np.all(schlieren_img == [126, 255, 130], axis=-1)  # Create a mask for matching pixels
            schlieren_img[mask] = [0, 0, 0]  # Set those pixels to black
            schlieren.append(schlieren_img)
            # print(schlieren_img)

            # Blend the original image with the Schlieren image
            overlay = cv2.addWeighted(img2, 1 - overlay_alpha, schlieren_img, overlay_alpha, 0)
            schlieren_frames.append(overlay)
    
    # only show red
    # schlieren = schlieren[:,,0]


    # average_image = np.mean(np.stack(schlieren, axis=0), axis=0).astype(np.uint8)
    # average_image = cv2.normalize(average_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # cv2.imshow("schlieren_avg", average_image)
    # cv2.waitKey(0)

    return schlieren_frames

        




def apply_schlieren_overlay(images, overlay_alpha=0.2):
    """
    Creates a Schlieren-like animation by visualizing changes between consecutive frames
    and overlaying them with the original images.
    """
    schlieren_frames = []
    schlieren = []
    
    for i in range(len(images) - 1):
        img2 = images[i + 1]
        img1 = images[i]

        img1_int = img1.astype(np.int16)
        img1_int -= 128

        img2_int = img2.astype(np.int16)
        img2_int -= 128

        scale_sch = 5

        gray_diff = cv2.subtract(img2_int, img1_int)*scale_sch
        gray_diff = np.clip(gray_diff, -127, 127)
        gray_diff[(gray_diff > -(2*scale_sch)) & (gray_diff < (2*scale_sch))] = 0
        


        grey_diff_uint8 = (gray_diff+128).astype(np.uint8)
        gray_diff_g = cv2.cvtColor(grey_diff_uint8, cv2.COLOR_BGR2GRAY)
        grey_diff_g = np.clip(gray_diff_g, 0, 255).astype(np.uint8)

        # Apply a colormap for visualization
        schlieren_img = cv2.applyColorMap(gray_diff_g, cv2.COLORMAP_JET)

        # Resize Schlieren image to match the original dimensions (if needed)
        schlieren_img = cv2.resize(schlieren_img, (img2.shape[1], img2.shape[0]))
        
        # add depth to the image set to 1 for now

        # find the mode colour
        # mode = np.argmax(np.bincount(schlieren_img.flatten()))

        # get rid of the mode colour setting it to black 
        # schlieren_img[schlieren_img == mode] 

        # if colour = 126 255 130 set it to 0 0 0 
        mask = np.all(schlieren_img == [126, 255, 130], axis=-1)  # Create a mask for matching pixels
        schlieren_img[mask] = [0, 0, 0]  # Set those pixels to black
        schlieren.append(schlieren_img)
        

        # print(schlieren_img)

        # Blend the original image with the Schlieren image
        overlay = cv2.addWeighted(img2, 1 - overlay_alpha, schlieren_img, overlay_alpha, 0)
        schlieren_frames.append(overlay)

    #average image of all schlieren images
    


    return schlieren_frames

    

# Example usage
coverage_id = "mtg_fd__vis06_hrfi"  # Example coverage ID
# bbox = [-3, 51, 0, 55]  # Bounding box for the UK
bbox = [17.5, -25, 20.5, -21]  # Bounding box for Namibia
start_time = "2024-11-14T12:01:00"  # Start of the time range
end_time = "2024-11-14T13:01:00"  # End of the time range
interval = 10  # Time interval in minutes

edsr_model = "lapSRN_x8.pb" 
# Step 1: Download images
images = download_images(coverage_id, bbox, start_time, end_time, interval_minutes=interval)

#append images to a list
images_1= []

for i in images:
    img = cv2.imread(i)

    upscaled_image = cv2.resize(img, 
                            None, 
                            fx=8, 
                            fy=8, 
                            interpolation=cv2.INTER_NEAREST)


    images_1.append(upscaled_image)


# if images:
    #    images = upscale_images(images, edsr_model, scale=8)

# Step 2: Interpolate frames and create smooth animation
if images:
    # images = list(reversed(images))  # Reverse the order of images
    # images = interpolate_frames_optical_flow(images, fps=30, interp_factor=30,upscale=8)
    # images = schlieren2(images, overlay_alpha=0.2,scale=8)
    # Step 3: Save the interpolated frames as a video
    frames_to_video(images_1, output_video="smooth_animation0.mp4", fps=1)