from tensorflow/tensorflow:1.15.5-gpu

RUN pip install absl-py==0.11.0 dm-sonnet==1.36 numpy


# This command runs your application, comment out this line to compile only
CMD ["bash"]
