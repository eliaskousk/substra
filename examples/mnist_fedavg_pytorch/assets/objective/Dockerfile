FROM eliaskousk/substra-tools:0.7.0-pytorch-slim

# add your metrics script to docker image
ADD metrics.py .

# define how script is run
ENTRYPOINT ["python3", "metrics.py"]
