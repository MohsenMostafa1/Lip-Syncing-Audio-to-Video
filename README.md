# Lip-Syncing-Audio-to-Video
<figure>
        <img src="https://ar5iv.labs.arxiv.org/html/2212.04970/assets/x1.png" alt ="Audio Art" style='width:800px;height:500px;'>
        <figcaption>


Automated Lip-Sync Video Generation: A Technical Overview for Clients    

This script demonstrates an automated pipeline for generating lip-synced videos  using the Wav2Lip  deep learning model. It seamlessly integrates audio and visual inputs to create realistic mouth movements that match a provided audio track, ideal for applications like content creation, dubbing, or virtual avatars. Below is a breakdown of how the system works:   
Key Steps in the Pipeline  

    Setup & Dependencies    
        Installs required libraries (librosa, ffmpeg, Wav2Lip) and downloads a pre-trained model (wav2lip_gan.pth).  
        Uses GPU acceleration (cuda) for faster inference.
         

    Input Requirements    
        Audio : A .wav file (e.g., 96_E.wav) containing the target speech.  
        Video : A .mp4 file (e.g., 13_K.mp4) featuring a face, which serves as the base for lip synchronization.
         

    Inference Process    
        The model processes the video frames and audio waveform to generate lip movements that align with the audio.  
        Handles memory constraints by dynamically adjusting batch sizes during computation (e.g., reducing batch size if GPU memory is exceeded).  
        Combines results into a final video (result_voice.mp4) using ffmpeg for audio-visual synchronization.
         

    Output Delivery    
        Saves the output video in two locations for easy access.  
        Embeds a playable preview of the video directly in the interface (if supported).
         
     

Why This Solution Works  

    High-Quality Results : Leverages the state-of-the-art Wav2Lip model, known for its accuracy in syncing speech with facial movements.  
    Robustness : Automatically recovers from out-of-memory errors, ensuring reliable execution even on limited hardware.  
    Efficiency : Takes ~7–8 minutes to process a 40-second clip, with GPU acceleration significantly reducing runtime.
     

Use Cases  

    Content Creation : Sync dubbed audio with existing video footage.  
    Virtual Presenters : Animate avatars or synthetic faces with realistic lip movements.  
    Accessibility : Generate subtitled or dubbed videos for diverse audiences.
     

This pipeline provides a scalable, automated way to create professional-grade lip-synced videos, saving time and effort compared to manual editing. Let us know if you’d like to adapt it for your specific use case!   


### Environment Setup
First, the necessary libraries and dependencies are installed. This includes librosa for audio processing, ffmpeg-python for handling video and audio files, and other dependencies required
 by the Wav2Lip model.
 
 ```python
!pip install librosa==0.9.1
!git clone https://github.com/zabique/Wav2Lip
!cd Wav2Lip && pip install -r requirements.txt
!pip install ffmpeg-python
!rm -rf /sample_data
!mkdir /sample_data
```

### Downloading Resources
Two critical resources are downloaded: a face detection model (s3fd.pth) and the Wav2Lip checkpoint file (wav2lip_gan.pth). These models are essential for the Wav2Lip inference process.

```python
!wget "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth" -O "/kaggle/working/Wav2Lip/face_detection/detection/sfd/s3fd.pth"

# Create checkpoints directory if it doesn't exist
!mkdir -p /kaggle/working/Wav2Lip/checkpoints
# Download the wav2lip_gan.pth checkpoint file
!wget "https://github.com/Rudrabha/Wav2Lip/releases/download/v0.1/wav2lip_gan.pth" -O "/kaggle/working/Wav2Lip/checkpoints/wav2lip_gan.pth"
```

### Verifying Checkpoint File
The script verifies the integrity of the downloaded checkpoint file by checking its size. This step ensures that the file is not corrupted or incomplete.

```python
import os
checkpoint_path = "/kaggle/working/Wav2Lip/checkpoints/wav2lip_gan.pth"
if os.path.exists(checkpoint_path):
    file_size = os.path.getsize(checkpoint_path)
    print(f"Checkpoint file size: {file_size} bytes")
    if file_size < 1000000:  # Replace with the expected size in bytes
        print("Checkpoint file seems to be corrupted or incomplete, re-downloading...")
        !wget "https://github.com/Rudrabha/Wav2Lip/releases/download/v0.1/wav2lip_gan.pth" -O "/kaggle/working/Wav2Lip/checkpoints/wav2lip_gan.pth"
else:
    print("Checkpoint file not found, downloading again...")
    !wget "https://github.com/Rudrabha/Wav2Lip/releases/download/v0.1/wav2lip_gan.pth" -O "/kaggle/working/Wav2Lip/checkpoints/wav2lip_gan.pth"
```

### Directory Listing
The script lists the contents of the relevant directories to verify the presence of necessary files before running the inference.

```python
print("Contents of /kaggle/working/Wav2Lip:")
!ls /kaggle/working/Wav2Lip
print("Contents of /kaggle/input/generative-ai:")
!ls /kaggle/input/generative-ai/
```
### Running Inference
The inference command is executed to synchronize the lip movements in the video (13_K.mp4) with the provided audio (96_E.wav). The output video is saved in the specified location.

```python
try:
    print("Running inference...")
    inference_output = !cd /kaggle/working/Wav2Lip && python inference.py --checkpoint_path checkpoints/wav2lip_gan.pth --face "/kaggle/input/generative-ai/13_K.mp4" --audio "/kaggle/input/generative-ai/96_E.wav" --outfile "results/result_voice.mp4"
    print("Inference output:")
    print("\n".join(inference_output))
    print("Inference completed successfully.")
except Exception as e:
    print(f"Inference failed: {e}")
```
### Post-Inference Checks
After the inference, the script lists the contents of the results directory to ensure the output file has been generated. If the result file exists, it is copied to a specified location for further use.

```python
import os
print("Contents of /kaggle/working/Wav2Lip/results after inference:")
!ls /kaggle/working/Wav2Lip/results

result_file = '/kaggle/working/Wav2Lip/results/result_voice.mp4'
if os.path.exists(result_file):
    output_file = '/kaggle/working/result_voice.mp4'
    import shutil
    shutil.copy(result_file, output_file)
    print(f"Result file saved as {output_file}")

    from IPython.display import HTML
    from base64 import b64encode

    mp4 = open(result_file,'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    display(HTML(f"""
    <video width="50%" height="50%" controls>
          <source src="{data_url}" type="video/mp4">
    </video>"""))
else:
    print(f"Result file {result_file} not found.")
```
