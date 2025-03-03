## Docker Build and Execution

### Build the Docker Image

1. **Navigate to Docker\_Image Folder**: Ensure you are in the directory containing the `Dockerfile` and project files.  
     
   `cd <path_to_folder>/Docker_Image`  
     
2. **Build the Docker Image**: Run the following command to build the Docker image.  
     
   `docker build -t <image_name>:<tag> .`

### Run the Docker Container

To run the Docker container from the image, use the following command:

`docker run --rm --gpus all -v "volume_path:/app/data" <image`  
`_name>:<tag> python3 app.py /app/data/input_file.json /app/data/Counts.json`

1. `--rm` removes the container after it exits.  
2. `--gpus all` ensures that all available GPUs are utilized.  
3. `-v "volume_path:/app/data"` mounts the local volume to the container’s `/app/data` directory.  
4. Replace `volume_path` with the path to your local data directory.  
5. `python3 app.py /app/data/input_file.json /app/data/Counts.json` runs the `app.py` script with specified input and output files.

## Notebooks and Scripts

1. **Docker\_Image Folder**: Contains the Dockerfile and main program files.  
   - `app.py` contains main program code  
   - `predict.py` contains code for prediction using XGBoost  
   - `config.json` contains coordinates for region polygons and required turning pattern for respective camera views  
   - `template.json` this json file is used as blank structure to format cumulative and prediction counts into appropriate “Counts.json” format  
   - `model.pt` custom trained YOLOv8s model

2. **`colab_training_notebook.ipynb`**: Used for training. This notebook uses YOLOv8s with default parameters, custom parameters include:  
     
   - `imgsz=1024` (sets image width to 1024 pixels)  
   - `epochs=25` (trains model for 25 epochs)

3. **Debugging Folder**: Contains code to annotate video with regions and bounding boxes.  
   - `Debugging/data` folder contains input and output files as well as videos  
   - run `debug.py` and get the annotated video in `data/output_vid.mp4`

## Dependencies

The required libraries and their versions are listed in `requirements.txt`.

Content of `requirements.txt`:

ultralytics==8.0.196  
opencv-python-headless==4.10.0.84  
lapx==0.5.9.post1  
numpy==2.0.1  
pandas==2.2.2  
xgboost==2.1.1  
scikit-learn==1.5.1

## Models Used

4. **YOLOv8s**: The project utilizes YOLOv8s trained on a custom labeled dataset. For more details, visit [Ultralytics YOLOv8 GitHub](https://github.com/ultralytics/yolov5).

## System Requirements

The project has been tested and evaluated on a workstation with the following specifications:

5. **CPU**: Intel Core i5-12th gen  
6. **RAM**: 16 GB  
7. **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU  
8. **Disk Space**: 512 GB SSD

Ensure your environment matches or exceeds these specifications for optimal performance.  
