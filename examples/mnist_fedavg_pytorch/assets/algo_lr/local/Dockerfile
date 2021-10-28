FROM eliaskousk/substra-tools:0.7.0-pytorch

# add your algorithm script to docker image
ADD algo_local.py .

# define how script is run
ENTRYPOINT ["python3", "algo_local.py"]
