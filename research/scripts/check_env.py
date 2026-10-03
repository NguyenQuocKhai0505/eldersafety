import sys, cv2, numpy, torch, ultralytics

print("Python     :", sys.version.split()[0])
print("OpenCV     :", cv2.__version__)
print("NumPy      :", numpy.__version__)
print("PyTorch    :", torch.__version__)
print("Ultralytics:", ultralytics.__version__)
print("GPU CUDA   :", torch.cuda.is_available())