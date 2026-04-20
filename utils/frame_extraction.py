import cv2
import os
from multiprocessing import Pool

def extract_frames(args):
    video_path, output_path_folder = args
    
    filename = os.path.basename(video_path)
    video_name = os.path.splitext(filename)[0]
    
    video_output_path = os.path.join(output_path_folder, video_name)
    
    if not os.path.exists(video_output_path):
        os.makedirs(video_output_path)
        
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 1
    
    while True:
        success, frame = cap.read()
        
        if not success:
            break
        
        cv2.imwrite(os.path.join(video_output_path, f"{video_name}_f{frame_count:05d}.jpg"), frame)
        
        frame_count += 1
    
    cap.release()
    return f"Video done: {filename}"

if __name__ == '__main__':
    input_dir = 'assets/Videos_Fitcow/VideosCurtos'
    output_dir = 'assets/Frames'
    
    valid_ext = ('.mp4', '.mov', '.MOV')
    
    tasks = [
        (os.path.join(input_dir, f), output_dir)
        for f in os.listdir(input_dir) if f.endswith(valid_ext)
    ]
    
    with Pool(processes=6) as pool:
        results = pool.map(extract_frames, tasks)
        print("\n".join(results))