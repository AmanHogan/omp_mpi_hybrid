# Using the script

Just run the script using the host files I have. I sample the output given running on the clusters in the project folder. Output will always be in output.txt

1. `chmod +x script.sh`
2. `./script.sh`

# Running individually

`mpicc -fopenmp 3dmpi_omp.c -o 3dmpi_omp -lm`

### Configuration a OMPX=1, OMPY=1, OMPZ=1, MPIX=2, MPIY=5, MPIZ=5

`mpirun -n 50 ./3dmpi_omp 2 5 5 1 1 1`

### Configuration b OMPX=1, OMPY=1, OMPZ=5, MPIX=2, MPIY=5, MPIZ=1

`mpirun -n 10 ./3dmpi_omp 2 5 1 1 1 5`

### Configuration c OMPX=1, OMPY=5, OMPZ=5, MPIX=2, MPIY=1, MPIZ=1

`mpirun -n 2 ./3dmpi_omp 2 1 1 1 5 5`

# Visualize
You can always use the visualize script like in project 4 and 3 if you want.