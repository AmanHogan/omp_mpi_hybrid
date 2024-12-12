#!/bin/bash

# Source file and executable
SRC="3dmpi_omp.c"     # Source file in current directory
EXE="3dmpi_omp"       # Executable in current directory
OUTPUT="output.txt"    # Output file in current directory
EXE_HOST="3dmpi_omp"  # Name of executable on remote hosts

# Host configuration
HOST1="192.168.5.132"
HOST2="192.168.5.133"
HOSTFILE="hosts"

# Configurations for MPI + OpenMP
# Format: "MPIX MPIY MPIZ OMPX OMPY OMPZ"
configs=(
    "2 5 5 1 1 1"    # Configuration a
    "2 5 1 1 1 5"    # Configuration b
    "2 1 1 1 5 5"    # Configuration c
)

# Clean up old files
rm -f $EXE $OUTPUT

echo "-------------------------------------------------------"
echo "Compiling the MPI + OpenMP program ($SRC)..."
mpicc -O3 -march=native -funroll-loops -fopenmp -o $EXE $SRC -lm
if [ $? -ne 0 ]; then
    echo "Error: Compilation failed."
    exit 1
fi

# Ensure executable is executable
chmod +x ./$EXE

# Function to copy the executable to specified nodes and set permissions
copy_to_nodes() {
    local program=$1
    echo "Copying $program to nodes $HOST1 and $HOST2..."
    
    # Copy to HOST1
    scp "./$program" "$HOST1:~/$program"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to copy $program to $HOST1"
    else
        ssh "$HOST1" "chmod +x ~/$program"
    fi
    
    # Copy to HOST2
    scp "./$program" "$HOST2:~/$program"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to copy $program to $HOST2"
    else
        ssh "$HOST2" "chmod +x ~/$program"
    fi
}

echo "-------------------------------------------------------"
echo "Running the hybrid MPI + OpenMP program..."

# Run each configuration
for config in "${configs[@]}"; do
    echo "-------------------------------------------------------"
    echo "Running configuration: $config"
    
    # Extract MPI and OMP values
    read MPIX MPIY MPIZ OMPX OMPY OMPZ <<< $config
    
    # Calculate total MPI processes and OMP threads
    MPI_TOTAL=$((MPIX * MPIY * MPIZ))
    OMP_THREADS=$((OMPX * OMPY * OMPZ))
    
    echo "Using $MPI_TOTAL MPI processes with $OMP_THREADS OpenMP threads each"
    
    # Copy executable to all nodes before each configuration
    copy_to_nodes "$EXE"
    
    # Set OpenMP environment variables
    export OMP_NUM_THREADS=$OMP_THREADS
    export OMP_PLACES=cores
    export OMP_PROC_BIND=close
    
    # Run with MPI
    mpirun --tag-output \
           --hostfile $HOSTFILE \
           -n $MPI_TOTAL \
           --map-by node:PE=$OMP_THREADS \
           --bind-to core \
           $EXE_HOST $MPIX $MPIY $MPIZ $OMPX $OMPY $OMPZ N=400 T=1.0 >> $OUTPUT
    
    echo "Configuration completed"
done

echo "All configurations completed. Output saved to $OUTPUT."
echo "-------------------------------------------------------"