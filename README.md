# deepracer-learning
# Setting Up a Local Training Environment for AWS DeepRacer on WSL2 (Detailed Guide)

## 1. Introduction

Training your AWS DeepRacer model locally using WSL2 offers several advantages:
- **Faster iteration cycles** - No need to wait for cloud uploads/downloads
- **Cost savings** - Avoid cloud computing charges during development
- **Offline capability** - Work anytime without internet connectivity
- **Full control** - Complete customization of training environments
- **GPU acceleration** - Leverage your powerful NVIDIA GPU directly

In this setup, we'll use:
- **WSL2**: Windows Subsystem for Linux 2 with full Linux kernel support
- **Docker Desktop**: Containerization platform for isolated training environments
- **NVIDIA Container Toolkit**: Enables GPU access within Docker containers
- **deepracer-for-cloud**: Open-source framework that mimics AWS DeepRacer cloud behavior locally

This approach gives you the same experience as cloud training but with local execution and maximum performance.

---

## 2. Prerequisites (Detailed)

### Step 1: Enable WSL2 on Windows

1. **Open PowerShell as Administrator**:
   - Press `Win + X` and select "Windows PowerShell (Admin)"
   - Or search for "PowerShell" in Start menu, right-click and choose "Run as administrator"

2. **Enable WSL2 features**:
   ```powershell
   # Enable required Windows features
   wsl --install -D Ubuntu-20.04
   
   # Alternative method if above doesn't work:
   dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
   dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
   ```

3. **Update WSL2 kernel**:
   ```powershell
   # Download and install the latest WSL2 Linux kernel update package
   # Visit: https://aka.ms/wsl2kernel
   # Run the downloaded .msi file
   ```

4. **Set WSL2 as default version**:
   ```powershell
   wsl --set-default-version 2
   ```

5. **Restart your computer** when prompted to complete installation.

> 🛠️ Verify WSL2 is working:
```powershell
wsl --list --verbose
```
You should see Ubuntu with version 2 listed.

---

### Step 2: Install a Linux Distribution (Ubuntu 20.04)

1. **Open Microsoft Store**:
   - Search for "Ubuntu" in the Microsoft Store
   - Select "Ubuntu 20.04 LTS" (or latest available)
   - Click "Get" and then "Install"

2. **Launch Ubuntu for first time setup**:
   - Click "Open" from the app list or search for "Ubuntu"
   - Wait for the installation to complete
   - Set up your username and password when prompted

3. **Update system packages**:
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```

---

### Step 3: Install Docker Desktop for Windows (Complete)

1. **Download Docker Desktop**:
   - Visit https://www.docker.com/products/docker-desktop/
   - Download "Docker Desktop for Windows (WSL 2 backend)"

2. **Install Docker Desktop**:
   - Run the downloaded installer
   - Follow the setup wizard
   - When prompted, select "Use WSL 2 based engine" and "Use the WSL 2 based engine"

3. **Enable WSL2 Backend in Docker Settings**:
   - Open Docker Desktop
   - Go to Settings → General
   - Check "Use the WSL 2 based engine"
   - Go to Settings → Resources → WSL Integration
   - Enable integration for your Ubuntu distribution

4. **Restart Docker Desktop** after making changes.

---

### Step 4: Configure NVIDIA GPU with Docker on WSL2 (CUDA 12.9 Compatible)

Since you have NVIDIA driver with CUDA 12.9, we'll ensure compatibility:

1. **Verify NVIDIA Driver Version in Windows**:
   - Open "NVIDIA Control Panel" or run `nvidia-smi` in Command Prompt
   - Confirm your driver version supports CUDA 12.9

2. **Install NVIDIA Container Toolkit on WSL2**:
   ```bash
   # Update package list
   sudo apt update
   
   # Install prerequisites
   sudo apt install -y curl gnupg ca-certificates software-properties-common
   
   # Add NVIDIA's GPG key
   curl -fsSL https://nvidia.github.io/nvidia-docker/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-archive-keyring.gpg
   
   # Add the repository
   echo "deb [signed-by=/usr/share/keyrings/nvidia-archive-keyring.gpg] https://nvidia.github.io/nvidia-docker/ubuntu20.04/amd64 stable" | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   
   # Update package list again
   sudo apt update
   
   # Install nvidia-container-toolkit
   sudo apt install -y nvidia-container-toolkit
   
   # For CUDA 12.9 support, also install the specific version
   sudo apt install -y nvidia-docker2
   ```

3. **Restart Docker daemon**:
   ```bash
   sudo systemctl restart docker
   ```

4. **Test GPU Access with Docker**:
   ```bash
   # Test basic GPU access
   docker run --gpus all nvidia/cuda:11.0-base-ubuntu20.04 nvidia-smi
   
   # For newer CUDA versions (if needed)
   docker run --gpus all nvidia/cuda:12.9.0-base-ubuntu20.04 nvidia-smi
   ```

5. **Verify NVIDIA Container Toolkit Installation**:
   ```bash
   # Check if the toolkit is properly installed
   nvidia-container-cli --version
   ```

> ⚠️ Note: If you encounter issues, try installing the latest nvidia-docker2 package that supports CUDA 12.9 specifically.

---

## 3. Setting Up the `deepracer-for-cloud` Environment

### Step 1: Clone the Repository

1. **Open your WSL terminal** (Ubuntu)
2. **Navigate to desired directory**:
   ```bash
   cd ~
   ```

3. **Clone the repository**:
   ```bash
   git clone https://github.com/aws-deepracer-community/deepracer-for-cloud.git
   cd deepracer-for-cloud
   ```

4. **Make sure you have Git installed** (if not):
   ```bash
   sudo apt install -y git
   ```

### Step 2: Run the Init Script with Detailed Steps

1. **Check script permissions**:
   ```bash
   ls -la init.sh
   ```

2. **Make script executable** (if needed):
   ```bash
   chmod +x init.sh
   ```

3. **Run initialization for GPU training in local mode**:
   ```bash
   ./init.sh -a gpu -c local
   ```

#### Explanation of Parameters:
- `-a gpu`: Enables GPU acceleration support
- `-c local`: Configures for local training instead of cloud simulation

### Step 3: Detailed Init Script Output Analysis

When you run `./init.sh -a gpu -c local`, you should see output like:

```
[INFO] Setting up local environment with GPU support...
[INFO] Downloading required Docker images...
[INFO] Starting MinIO container for local S3 compatibility...
[INFO] Configuring NVIDIA Container Toolkit...
[INFO] Environment ready for training!
```

### Step 4: Verify Init Success

1. **Check running containers**:
   ```bash
   docker ps -a
   ```

2. **Expected containers should include**:
   - MinIO (local S3)
   - RoboMaker simulator
   - Training container

3. **Test if MinIO is accessible**:
   ```bash
   curl http://localhost:9000/minio/health/live
   ```

---

## 4. Configuring Your DeepRacer Model (Detailed)

### Step 1: Create Model Directory Structure

1. **Navigate to models directory**:
   ```bash
   cd ~/deepracer-for-cloud/models
   ```

2. **Create your model folder**:
   ```bash
   mkdir my_custom_model
   cd my_custom_model
   ```

3. **Create required files**:
   ```bash
   touch reward_function.py
   touch model_metadata.json
   touch hyperparameters.json
   ```

### Step 2: Place Custom Files in Correct Locations

#### Example `reward_function.py`:
```python
def reward_function(params):
    """
    Reward function for DeepRacer training
    """
    # Example reward logic
    reward = 1.0
    
    # Add rewards for staying on track
    if params["is_on_track"]:
        reward += 1.0
    
    # Add penalty for going off track
    if not params["is_on_track"]:
        reward -= 1.0
        
    return float(reward)
```

#### Example `model_metadata.json`:
```json
{
  "model_name": "my_custom_model",
  "description": "Custom model for local training",
  "version": "1.0.0"
}
```

#### Example `hyperparameters.json`:
```json
{
  "learning_rate": 0.001,
  "batch_size": 32,
  "epochs": 100,
  "gamma": 0.99,
  "epsilon_start": 1.0,
  "epsilon_end": 0.01,
  "epsilon_decay": 0.995
}
```

### Step 3: Upload Custom Files to Local S3 (MinIO)

1. **Return to project root**:
   ```bash
   cd ~/deepracer-for-cloud
   ```

2. **Upload your custom files**:
   ```bash
   dr-upload-custom-files -m my_custom_model
   ```

3. **Verify upload success**:
   ```bash
   # Check if files are stored in MinIO
   docker exec -it minio_container ls /data/my_custom_model/
   ```

> 📁 Important: Make sure the model name exactly matches what's specified in your `model_metadata.json` file.

---

## 5. Training Your Model (Detailed Steps)

### Step 1: Start Training Process

```bash
dr-start-training -m my_custom_model
```

### Step 2: Monitor Training Progress

1. **Watch training logs**:
   ```bash
   # View real-time logs
   dr-logs-robomaker
   
   # Or tail the logs in another terminal
   docker logs -f <training_container_name>
   ```

2. **Expected output during training**:
   ```
   [INFO] Starting episode 1...
   [INFO] Reward: 2.5
   [INFO] Lap time: 12.3s
   [INFO] Episode completed successfully
   ```

### Step 3: Training Metrics to Monitor

- **Reward scores**: Higher is better
- **Lap times**: Lower is better (faster completion)
- **Episode count**: Number of training episodes completed
- **Loss values**: For neural network optimization

---

## 6. Visualizing and Analyzing Your Training (Optional but Recommended)

### Step 1: Launch Live Viewer

```bash
dr-start-viewer
```

> 🎥 If you don't see the viewer window, try:
```bash
# Run with explicit display settings
DISPLAY=:0 dr-start-viewer
```

### Step 2: Log Analysis with Jupyter Notebooks

```bash
dr-start-loganalysis
```

This opens a Jupyter notebook where you can:
- Analyze reward trends over episodes
- Plot lap times and performance metrics
- Export training data for further analysis

### Step 3: View RoboMaker Logs

```bash
# Get logs from the RoboMaker container
dr-logs-robomaker

# Or check specific log files
docker logs robomaker_container_name
```

---

## 7. Troubleshooting Common Issues (Detailed Solutions)

### Issue 1: Docker Not Starting Correctly with Permission Errors

**Error Message**: `Got permission denied while trying to connect to the Docker daemon socket`

**Solution Steps**:
1. **Add user to docker group**:
   ```bash
   sudo usermod -aG docker $USER
   ```

2. **Reboot or re-login** (or run):
   ```bash
   newgrp docker
   ```

3. **Verify Docker access**:
   ```bash
   docker ps
   ```

### Issue 2: File Path Errors When Uploading Custom Files

**Error Message**: `No such file or directory` when uploading files

**Solution Steps**:
1. **Check current working directory**:
   ```bash
   pwd
   ls -la
   ```

2. **Verify model directory exists**:
   ```bash
   ls ~/deepracer-for-cloud/models/my_custom_model/
   ```

3. **Ensure all files are present**:
   ```bash
   ls -la ~/deepracer-for-cloud/models/my_custom_model/*.py
   ls -la ~/deepracer-for-cloud/models/my_custom_model/*.json
   ```

4. **Use absolute paths if needed**:
   ```bash
   dr-upload-custom-files -m /home/$(whoami)/deepracer-for-cloud/models/my_custom_model
   ```

### Issue 3: Video Stream Viewer Doesn't Work

**Problem**: No live feed in viewer window

**Solution Steps**:
1. **Ensure training is running first**:
   ```bash
   # Check if training container is active
   docker ps | grep training
   ```

2. **Install WSLg support if missing**:
   ```bash
   sudo apt install -y wslu
   ```

3. **Try launching viewer with different display settings**:
   ```bash
   export DISPLAY=:0
   dr-start-viewer
   ```

### Issue 4: GPU Drivers Not Working Properly

**Error**: `Failed to initialize NVIDIA context` or similar GPU errors

**Solution Steps**:
1. **Verify NVIDIA driver compatibility**:
   ```bash
   nvidia-smi
   ```

2. **Check if Docker can see GPUs**:
   ```bash
   docker run --gpus all nvidia/cuda:12.9.0-base-ubuntu20.04 nvidia-smi
   ```

3. **Reinstall NVIDIA Container Toolkit**:
   ```bash
   sudo apt remove -y nvidia-container-toolkit nvidia-docker2
   sudo apt autoremove
   # Then reinstall following steps in Step 4 above
   ```

4. **Restart Docker daemon**:
   ```bash
   sudo systemctl restart docker
   ```

5. **Check WSL2 GPU support**:
   ```bash
   cat /proc/driver/nvidia/version
   ```

---

## 8. Conclusion

You now have a fully functional local AWS DeepRacer training environment with the following capabilities:

### Key Features Implemented:
✅ **WSL2 Integration** - Native Linux subsystem on Windows  
✅ **Docker Containerization** - Isolated training environments  
✅ **NVIDIA GPU Acceleration** - CUDA 12.9 compatible  
✅ **Local S3 Storage** - MinIO for file management  
✅ **Training Monitoring** - Real-time progress tracking  
✅ **Visualization Tools** - Live viewer and log analysis  

### Next Steps to Explore:
- **Experiment with reward functions** - Create more complex reward logic
- **Tune hyperparameters** - Adjust learning rates, batch sizes, etc.
- **Multiple model comparisons** - Train several variations side-by-side
- **Advanced analytics** - Use Jupyter notebooks for deep performance analysis

### Additional Resources:
- [deepracer-for-cloud GitHub Repository](https://github.com/aws-deepracer-community/deepracer-for-cloud)
- [AWS DeepRacer Documentation](https://docs.aws.amazon.com/deepracer/)
- [NVIDIA Container Toolkit Docs](https://github.com/NVIDIA/nvidia-docker)

Happy training! 🏁🚀 Your local environment is ready to help you build better racing models faster than ever before.


Ver#2
====================================================================================================================================================================================================

# Setting Up a Local Training Environment for AWS DeepRacer on WSL2 (Complete Guide with CUDA 12.9.0)

## 1. Introduction

Training your AWS DeepRacer model locally using WSL2 with CUDA 12.9.0 provides the ultimate development experience:
- **Maximum Performance**: Leverage the latest NVIDIA CUDA capabilities
- **Enhanced Compatibility**: Full support for modern GPU architectures
- **Seamless Integration**: Native Windows + Linux workflow
- **Cost Efficiency**: Eliminate cloud training costs during development
- **Rapid Iteration**: Fast feedback loops for model refinement

This comprehensive guide covers everything from system preparation to advanced troubleshooting specifically optimized for CUDA 12.9.0 compatibility.

---

## 2. Prerequisites (Complete & Detailed)

### Step 1: Enable WSL2 on Windows (Complete Verification)

1. **Open PowerShell as Administrator**:
   ```powershell
   # Run in elevated PowerShell
   Start-Process powershell -Verb RunAs
   ```

2. **Enable required Windows features**:
   ```powershell
   # Check if WSL is already installed
   wsl --list --verbose
   
   # Enable WSL features (if not enabled)
   dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
   dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
   
   # Install WSL2 kernel update if needed
   wsl --update
   ```

3. **Set WSL2 as default version**:
   ```powershell
   wsl --set-default-version 2
   ```

4. **Verify WSL2 installation**:
   ```powershell
   # Check installed versions
   wsl --list --verbose
   
   # Expected output should show:
   # NAME      FRIENDLY NAME    VERSION
   # Ubuntu    Ubuntu           2
   ```

5. **Restart your computer** to complete WSL2 installation.

### Step 2: Install Ubuntu 20.04 LTS Distribution

1. **Download and install Ubuntu from Microsoft Store**:
   - Open Microsoft Store
   - Search for "Ubuntu 20.04 LTS"
   - Click "Get" and then "Install"
   - Launch the app once to complete setup

2. **Complete initial setup**:
   ```bash
   # Set username and password when prompted
   # This typically happens automatically on first launch
   
   # Update system packages immediately
   sudo apt update && sudo apt upgrade -y
   ```

3. **Install essential tools**:
   ```bash
   sudo apt install -y build-essential curl wget git vim htop
   ```

### Step 3: Install Docker Desktop for Windows (Complete Setup)

1. **Download Docker Desktop**:
   - Visit https://www.docker.com/products/docker-desktop/
   - Download "Docker Desktop for Windows (WSL 2 backend)"
   - Run installer with default settings
   - 

2. **Configure Docker Desktop WSL2 Integration**:
   ```powershell
   # Open Docker Desktop Settings
   # Go to Settings → General
   # Ensure "Use the WSL 2 based engine" is checked
   
   # Go to Settings → Resources → WSL Integration
   # Enable integration for your Ubuntu distribution
   ```
   You can also install docker directly
   ```bash
      curl https://get.docker.com | sh
      sudo service docker start
   ```

4. **Restart Docker Desktop** after configuration changes

5. **Verify Docker installation in WSL2**:
   ```bash
   # In Ubuntu terminal
   docker --version
   docker info
   
   # Test basic container
   docker run hello-world
   ```

### Step 4: Configure NVIDIA GPU with CUDA 12.9.0 Support

#### 4.1 Verify NVIDIA Driver Compatibility

1. **Check Windows NVIDIA driver version**:
   ```powershell
   # Open Command Prompt and run:
   nvidia-smi
   
   # Expected output should show CUDA version >= 12.9.0
   ```

2. **Download NVIDIA CUDA Toolkit 12.9.0**:
   - Visit https://developer.nvidia.com/cuda-12-9-0-download-archive
   - Select "Linux" → "x86_64" → "Ubuntu" → "20.04"
   - Download the .deb package

3. **Install CUDA Toolkit 12.9.0 in WSL2**:
   ```bash
   # Navigate to download directory
   cd ~/Downloads
   
   # Install CUDA toolkit
   sudo dpkg -i cuda-toolkit-12-9_12.9.0-1_amd64.deb
   sudo apt update
   
   # Install additional dependencies
   sudo apt install -y libcudnn8 libcudnn8-dev
   
   # Verify installation
   nvcc --version
   ```

#### 4.2 Install NVIDIA Container Toolkit for CUDA 12.9.0

1. **Remove previous NVIDIA container toolkit**:
   ```bash
   sudo apt remove -y nvidia-container-toolkit nvidia-docker2
   sudo apt autoremove
   ```

2. **Add NVIDIA repository for latest toolkit**:
   ```bash
   # Add NVIDIA GPG key
   curl -fsSL https://nvidia.github.io/nvidia-docker/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-archive-keyring.gpg
   
   # Add repository for CUDA 12.9 support
   echo "deb [signed-by=/usr/share/keyrings/nvidia-archive-keyring.gpg] https://nvidia.github.io/nvidia-docker/ubuntu20.04/amd64 stable" | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   
   # Update package list
   sudo apt update
   ```

3. **Install NVIDIA Container Toolkit**:
   ```bash
   # Install latest compatible versions
   sudo apt install -y nvidia-container-toolkit nvidia-docker2
   
   # Verify installation
   nvidia-container-cli --version
   ```

4. **Restart Docker daemon**:
   ```bash
   sudo systemctl restart docker
   ```

5. **Test GPU access with CUDA 12.9.0 containers**:
   ```bash
   # Test basic GPU detection
   docker run --gpus all nvidia/cuda:12.9.0-base-ubuntu20.04 nvidia-smi
   
   # Test CUDA compilation capability
   docker run --gpus all nvidia/cuda:12.9.0-devel-ubuntu20.04 nvcc --version
   ```

6. **Verify complete CUDA 12.9.0 setup**:
   ```bash
   # Check CUDA installation in WSL2
   cat /usr/local/cuda/version.txt
   
   # Verify NVIDIA driver compatibility
   nvidia-smi -q | grep "CUDA Version"
   ```

---

## 3. Setting Up the `deepracer-for-cloud` Environment (Complete)

### Step 1: Clone and Prepare Repository

1. **Navigate to home directory**:
   ```bash
   cd ~
   ```

2. **Clone repository with full history**:
   ```bash
   git clone https://github.com/aws-deepracer-community/deepracer-for-cloud.git
   cd deepracer-for-cloud
   
   # Check repository status
   git status
   git log --oneline -5
   ```

3. **Install required dependencies**:
   ```bash
   # Install Python dependencies if needed
   sudo apt install -y python3-pip python3-dev
   
   # Install Python packages for the project
   pip3 install -r requirements.txt 2>/dev/null || echo "Requirements file may not exist"
   ```

### Step 2: Initialize Environment with CUDA 12.9.0 Support

1. **Make init script executable**:
   ```bash
   chmod +x init.sh
   ```

2. **Run initialization for CUDA 12.9.0 GPU training**:
   ```bash
   ./init.sh -a gpu -c local
   ```

3. **Monitor initialization progress**:
   ```bash
   # You should see output like:
   # [INFO] Setting up local environment with GPU support...
   # [INFO] Downloading required Docker images (may take several minutes)...
   # [INFO] Starting MinIO container for local S3 compatibility...
   # [INFO] Configuring NVIDIA Container Toolkit...
   # [INFO] Environment ready for training!
   ```

4. **Verify successful initialization**:
   ```bash
   # Check running containers
   docker ps -a
   
   # Expected containers:
   # - minio (local S3)
   # - robomaker (training environment)
   # - any other required services
   
   # Test MinIO connectivity
   curl http://localhost:9000/minio/health/live
   ```

### Step 3: Environment Configuration Verification

1. **Check NVIDIA GPU access**:
   ```bash
   # Verify Docker can access GPUs
   docker run --gpus all nvidia/cuda:12.9.0-base-ubuntu20.04 nvidia-smi
   
   # Check if CUDA libraries are accessible
   docker run --gpus all nvidia/cuda:12.9.0-base-ubuntu20.04 ls /usr/local/cuda/lib64/
   ```

2. **Verify Docker GPU configuration**:
   ```bash
   # Check Docker daemon config for GPU support
   cat /etc/docker/daemon.json
   
   # Should contain something like:
   {
     "default-runtime": "nvidia",
     "runtimes": {
       "nvidia": {
         "path": "nvidia-container-runtime",
         "runtimeArgs": []
       }
     }
   }
   ```

---

## 4. Configuring Your DeepRacer Model (Complete Setup)

### Step 1: Create Complete Model Directory Structure

1. **Navigate to models directory**:
   ```bash
   cd ~/deepracer-for-cloud/models
   ```

2. **Create comprehensive model structure**:
   ```bash
   mkdir -p my_cuda_129_model/{data,logs,checkpoints,saved_models}
   cd my_cuda_129_model
   
   # Create all required files
   touch reward_function.py
   touch model_metadata.json
   touch hyperparameters.json
   touch model_config.yaml
   ```

### Step 2: Implement Detailed Model Files

#### Example `reward_function.py` (CUDA-optimized):
```python
import math

def reward_function(params):
    """
    Advanced reward function optimized for CUDA 12.9.0 training
    Features:
    - Gradient-based rewards
    - Multi-factor optimization
    - GPU-efficient calculations
    """
    
    # Read parameters
    all_wheels_on_track = params['all_wheels_on_track']
    speed = params['speed']
    steering_angle = abs(params['steering_angle'])
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']
    is_offtrack = not params['is_on_track']
    
    # Base reward
    reward = 1.0
    
    # Reward for staying on track
    if all_wheels_on_track:
        reward += 1.5
    else:
        reward -= 2.0
    
    # Reward for speed (optimized for GPU computation)
    if speed > 3.5:  # High-speed threshold
        reward += 1.0
    elif speed > 2.5:  # Medium-speed threshold
        reward += 0.5
    else:
        reward -= 0.5
    
    # Penalty for sharp turns (GPU-efficient)
    if steering_angle > 30:
        reward -= 0.8
    elif steering_angle > 15:
        reward -= 0.3
    
    # Centerline reward (using math operations optimized for CUDA)
    reward += max(0, 1 - (distance_from_center / (track_width/2)))
    
    # Penalty for being off track
    if is_offtrack:
        reward = 1e-3  # Very small reward to penalize off-track
    
    # Ensure reward is within reasonable bounds
    reward = max(0.01, min(reward, 10.0))
    
    return float(reward)
```

#### Example `model_metadata.json`:
```json
{
  "model_name": "my_cuda_129_model",
  "description": "DeepRacer model optimized for CUDA 12.9.0 training environment",
  "version": "2.0.0",
  "author": "Local Training Setup",
  "created_date": "2024-01-15",
  "framework": "PyTorch",
  "cuda_version": "12.9.0",
  "gpu_acceleration": true,
  "training_environment": "local_wsl2"
}
```

#### Example `hyperparameters.json`:
```json
{
  "learning_rate": 0.001,
  "batch_size": 64,
  "epochs": 200,
  "gamma": 0.99,
  "epsilon_start": 1.0,
  "epsilon_end": 0.01,
  "epsilon_decay": 0.995,
  "max_steps_per_episode": 1000,
  "buffer_size": 100000,
  "learning_frequency": 4,
  "target_update_frequency": 1000,
  "cuda_device": "cuda:0",
  "optimizer": "Adam",
  "loss_function": "MSE",
  "dropout_rate": 0.3,
  "weight_decay": 0.0001
}
```

#### Example `model_config.yaml`:
```yaml
# Model configuration optimized for CUDA 12.9.0
model:
  name: "deep_racer_cuda_129"
  architecture: "DQN"
  input_shape: [3, 84, 84]
  output_size: 5
  hidden_layers: [512, 256, 128]
  activation: "relu"
  batch_norm: true

training:
  device: "cuda:0"
  num_epochs: 200
  batch_size: 64
  learning_rate: 0.001
  optimizer: "Adam"
  scheduler: "StepLR"
  gamma: 0.99
  epsilon_start: 1.0
  epsilon_end: 0.01
  epsilon_decay: 0.995

environment:
  render_mode: "human"
  max_steps_per_episode: 1000
  track_name: "circuit_1"
  reward_function: "reward_function.py"

hardware:
  gpu_memory_limit: "80%"
  cuda_version: "12.9.0"
  compute_capability: "7.5"
```

### Step 3: Upload Custom Files to Local S3 (MinIO)

1. **Return to project root**:
   ```bash
   cd ~/deepracer-for-cloud
   ```

2. **Verify model directory structure**:
   ```bash
   ls -la models/my_cuda_129_model/
   
   # Should show:
   # reward_function.py
   # model_metadata.json
   # hyperparameters.json
   # model_config.yaml
   ```

3. **Upload custom files to MinIO**:
   ```bash
   dr-upload-custom-files -m my_cuda_129_model
   
   # Monitor upload progress
   # You should see output confirming file transfers
   ```

4. **Verify successful upload**:
   ```bash
   # Check if files are stored in MinIO
   docker exec -it minio_container ls /data/my_cuda_129_model/
   
   # Expected output should list all your uploaded files
   ```

---

## 5. Training Your Model (Complete Execution)

### Step 1: Start Training Process with CUDA 12.9.0

```bash
# Start training with detailed logging
dr-start-training -m my_cuda_129_model --verbose

# Alternative with specific parameters
dr-start-training -m my_cuda_129_model \
  --epochs 200 \
  --batch-size 64 \
  --learning-rate 0.001 \
  --cuda-device cuda:0
```

### Step 2: Monitor Training Progress (Comprehensive)

1. **View real-time training logs**:
   ```bash
   # Real-time monitoring
   dr-logs-robomaker
   
   # Or in separate terminal
   docker logs -f robomaker_container_name
   ```

2. **Monitor GPU usage during training**:
   ```bash
   # Watch GPU utilization in real-time
   watch -n 1 nvidia-smi
   
   # Or more detailed monitoring
   nvidia-smi pmon -c 10
   ```

3. **Check system resources**:
   ```bash
   # Monitor CPU and memory usage
   htop
   
   # Check disk space
   df -h
   ```

### Step 3: Training Metrics Monitoring

1. **Expected training output patterns**:
   ```
   [INFO] Starting episode 1...
   [INFO] Reward: 2.8
   [INFO] Lap time: 15.2s
   [INFO] GPU Utilization: 75%
   [INFO] Memory Usage: 4.2GB / 16GB
   [INFO] Episode completed successfully
   
   [INFO] Training progress: 10/200 epochs (5%)
   [INFO] Average reward: 2.3
   [INFO] Best lap time: 12.1s
   ```

### Step 4: Advanced Training Controls

1. **Training with checkpoint saving**:
   ```bash
   dr-start-training -m my_cuda_129_model \
     --save-checkpoints \
     --checkpoint-interval 50 \
     --early-stopping-patience 20
   ```

2. **Training with custom parameters**:
   ```bash
   dr-start-training -m my_cuda_129_model \
     --max-episodes 300 \
     --learning-rate 0.0005 \
     --batch-size 32 \
     --gamma 0.98
   ```

---

## 6. Visualizing and Analyzing Your Training (Complete)

### Step 1: Launch Live Viewer with CUDA Optimization

```bash
# Start live viewer for real-time observation
dr-start-viewer

# Alternative with specific display settings
export DISPLAY=:0
dr-start-viewer --resolution 1280x720

# For headless environments
dr-start-viewer --headless
```

### Step 2: Log Analysis with Jupyter Notebooks

```bash
# Launch log analysis environment
dr-start-loganalysis

# This opens Jupyter notebook where you can:
# - Analyze reward trends over episodes
# - Plot lap times and performance metrics
# - Export training data for further analysis
```

### Step 3: Advanced Log Monitoring

1. **View detailed RoboMaker logs**:
   ```bash
   # Get comprehensive logs
   dr-logs-robomaker --tail 100
   
   # Filter specific log levels
   dr-logs-robomaker --level INFO
   
   # Save logs to file for later analysis
   dr-logs-robomaker > training_logs.txt
   ```

2. **Monitor GPU-specific logs**:
   ```bash
   # Check CUDA operations logs
   docker logs robomaker_container_name 2>&1 | grep -i cuda
   
   # Monitor memory usage in logs
   docker logs robomaker_container_name 2>&1 | grep -i memory
   ```

---

## 7. Troubleshooting Common Issues (Complete Guide)

### Issue 1: Docker GPU Access Not Working

**Symptom**: `docker: Error response from daemon: could not select device driver "" with capabilities: [gpu]`

**Solution Steps**:
1. **Verify NVIDIA Container Toolkit installation**:
   ```bash
   # Check toolkit version
   nvidia-container-cli --version
   
   # Verify Docker daemon configuration
   cat /etc/docker/daemon.json
   ```

2. **Reinstall Docker with proper GPU support**:
   ```bash
   # Remove existing Docker
   sudo apt remove -y docker-ce docker-ce-cli containerd.io
   
   # Install Docker with repository
   sudo apt update
   sudo apt install -y apt-transport-https ca-certificates curl gnupg lsb-release
   
   # Add Docker's official GPG key
   curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
   
   # Add repository
   echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
   
   # Install Docker
   sudo apt update
   sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
   
   # Configure NVIDIA runtime
   sudo mkdir -p /etc/docker
   sudo tee /etc/docker/daemon.json << EOF
   {
     "default-runtime": "nvidia",
     "runtimes": {
       "nvidia": {
         "path": "nvidia-container-runtime",
         "runtimeArgs": []
       }
     }
   }
   EOF
   
   # Restart Docker
   sudo systemctl restart docker
   ```

### Issue 2: CUDA 12.9.0 Compatibility Problems

**Symptom**: `CUDA error: no kernel image is available for execution on the device`

**Solution Steps**:
1. **Check CUDA compute capability compatibility**:
   ```bash
   # Check your GPU compute capability
   nvidia-smi -q | grep "Compute Capability"
   
   # Common capabilities:
   # RTX 30xx series: 8.6
   # RTX 40xx series: 8.9
   ```

2. **Reinstall CUDA toolkit for correct architecture**:
   ```bash
   # Remove existing installation
   sudo apt remove -y cuda-toolkit-12-9
   
   # Reinstall with proper flags
   sudo apt install -y cuda-toolkit-12-9 libcudnn8 libcudnn8-dev
   
   # Verify compute capability support
   nvcc --version
   ```

3. **Use appropriate Docker image tags**:
   ```bash
   # For CUDA 12.9 with specific compute capability
   docker run --gpus all nvidia/cuda:12.9.0-devel-ubuntu20.04 nvidia-smi
   ```

### Issue 3: File Path Errors During Upload

**Symptom**: `FileNotFoundError` or `Permission denied` when uploading files

**Solution Steps**:
1. **Check directory permissions**:
   ```bash
   # Verify model directory permissions
   ls -la ~/deepracer-for-cloud/models/my_cuda_129_model/
   
   # Fix permissions if needed
   chmod 755 ~/deepracer-for-cloud/models/my_cuda_129_model/
   ```

2. **Use absolute paths for uploads**:
   ```bash
   # Upload using full path
   dr-upload-custom-files -m /home/$(whoami)/deepracer-for-cloud/models/my_cuda_129_model
   
   # Or navigate to correct directory first
   cd ~/deepracer-for-cloud/models/my_cuda_129_model/
   dr-upload-custom-files -m .
   ```

### Issue 4: Video Stream Viewer Issues

**Symptom**: No video feed or viewer crashes

**Solution Steps**:
1. **Install required display libraries**:
   ```bash
   # Install X11 and GUI libraries
   sudo apt install -y x11-xserver-utils libgl1-mesa-glx libglu1-mesa
   
   # Install WSLg components if missing
   sudo apt install -y wslu
   ```

2. **Configure display environment**:
   ```bash
   # Set proper display variables
   export DISPLAY=:0
   export LIBGL_ALWAYS_INDIRECT=1
   
   # Test viewer with debug output
   dr-start-viewer --debug
   ```

3. **Alternative headless viewing**:
   ```bash
   # Start viewer without GUI for debugging
   dr-start-viewer --headless --verbose
   ```

### Issue 5: Memory Issues During Training

**Symptom**: Out of memory errors or slow training performance

**Solution Steps**:
1. **Monitor system resources during training**:
   ```bash
   # Watch memory usage in real-time
   watch -n 2 free -h
   
   # Monitor GPU memory specifically
   watch -n 2 nvidia-smi
   ```

2. **Adjust training parameters for memory efficiency**:
   ```bash
   # Reduce batch size and memory footprint
   dr-start-training -m my_cuda_129_model \
     --batch-size 32 \
     --max-episodes 100 \
     --learning-rate 0.0005
   ```

3. **Optimize model architecture for memory**:
   ```bash
   # Modify hyperparameters.json to reduce model complexity
   {
     "batch_size": 32,
     "epochs": 100,
     "hidden_layers": [256, 128],
     "dropout_rate": 0.2
   }
   ```

---

## 8. Advanced Performance Optimization

### GPU Memory Management

```bash
# Set GPU memory limit for training containers
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Monitor GPU memory usage during training
nvidia-smi -q -d MEMORY -l 1
```

### Training Script Optimization

```bash
# Create optimized training script
cat > train_optimized.sh << 'EOF'
#!/bin/bash

# Optimized training with CUDA 12.9.0
echo "Starting optimized training with CUDA 12.9.0..."

# Set environment variables for performance
export CUDA_LAUNCH_BLOCKING=0
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Start training
dr-start-training -m my_cuda_129_model \
  --epochs 200 \
  --batch-size 64 \
  --learning-rate 0.001 \
  --cuda-device cuda:0 \
  --verbose

echo "Training completed successfully!"
EOF

chmod +x train_optimized.sh
```

### Performance Monitoring Dashboard

```bash
# Install monitoring tools
sudo apt install -y htop iotop glances

# Launch performance dashboard during training
glances --time=5 --disable-update
```

---

## 9. Conclusion

You have now successfully configured a complete local AWS DeepRacer training environment with:

✅ **CUDA 12.9.0 Support** - Latest NVIDIA compute capabilities  
✅ **WSL2 Integration** - Seamless Windows + Linux workflow  
✅ **GPU Acceleration** - Full CUDA 12.9.0 GPU utilization  
✅ **Complete Toolchain** - Docker, MinIO, and training utilities  
✅ **Advanced Monitoring** - Real-time performance tracking  

### Key Features Achieved:
- **Maximum Performance**: Leveraging latest CUDA 12.9.0 optimizations
- **Reliable Training**: Stable GPU container environments
- **Comprehensive Monitoring**: Real-time system and training metrics
- **Robust Error Handling**: Comprehensive troubleshooting procedures

### Next Steps for Your Development Journey:

1. **Experiment with reward functions** - Implement complex racing strategies
2. **Tune hyperparameters** - Optimize for your specific track conditions
3. **Multi-model comparison** - Train several variations simultaneously
4. **Advanced analytics** - Use Jupyter notebooks for deep performance insights

### Additional Resources:
- [deepracer-for-cloud GitHub Repository](https://github.com/aws-deepracer-community/deepracer-for-cloud)
- [NVIDIA CUDA Documentation 12.9.0](https://docs.nvidia.com/cuda/)
- [AWS DeepRacer Developer Guide](https://docs.aws.amazon.com/deepracer/)

Happy training! 🏁🚀 Your powerful CUDA 12.9.0 optimized local environment is ready to help you build the best racing models possible!

