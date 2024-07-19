# Lip-Syncing-Audio-to-Video

### Environment Setup
First, the necessary libraries and dependencies are installed. This includes librosa for audio processing, ffmpeg-python for handling video and audio files, and other dependencies required
 by the Wav2Lip model.
 

### Downloading Resources
Two critical resources are downloaded: a face detection model (s3fd.pth) and the Wav2Lip checkpoint file (wav2lip_gan.pth). These models are essential for the Wav2Lip inference process.

#### Create checkpoints directory if it doesn't exist

#### Download the wav2lip_gan.pth checkpoint file

### Verifying Checkpoint File
The script verifies the integrity of the downloaded checkpoint file by checking its size. This step ensures that the file is not corrupted or incomplete.

### Directory Listing
The script lists the contents of the relevant directories to verify the presence of necessary files before running the inference.

### Running Inference
The inference command is executed to synchronize the lip movements in the video (13_K.mp4) with the provided audio (96_E.wav). The output video is saved in the specified location.

### Post-Inference Checks
After the inference, the script lists the contents of the results directory to ensure the output file has been generated. If the result file exists, it is copied to a specified location for further use.
