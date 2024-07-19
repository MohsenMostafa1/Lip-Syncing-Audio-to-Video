# Lip-Syncing-Audio-to-Video
<div align="center">
  <img src="[https://www.tensorflow.org/images/tf_logo_horizontal.png](https://camo.githubusercontent.com/3c044b086f1050830f1e624833f3d511037dbb6c471136b3a18a95ea6ca27b43/68747470733a2f2f6f70656e74616c6b65722e6769746875622e696f2f766964656f2d726574616c6b696e672f7374617469632f696d616765732f7465617365722e706e67)">
</div>

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
