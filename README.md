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
