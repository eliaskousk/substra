FROM eliaskousk/substra-tools:0.7.0-pytorch-slim

# add your algorithm script to docker image
ADD algo_global.py .

# define how script is run
ENTRYPOINT ["python3", "algo_global.py"]
