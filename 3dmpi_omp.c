/**
 * Author: Aman Hogan-Bailey
 * Solves 3D heat equation using hybrid MPI + OpenMP
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <time.h>

#define GRID_SIZE 80
#define TIME_FINAL 1.0
#define PI 3.14159265358979323846

// Indexing for 3D heat solver using MPI, accounting for ghost cells
#define I3DMPI(i, j, k, nx, ny) ((i) + (j) * (nx + 2) + (k) * (nx + 2) * (ny + 2))

int main(int argc, char **argv) 
{
    int provided;

    // Initialize MPI with thread support
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Default values for MPI and OpenMP splits
    int MPIX = 2, MPIY = 5, MPIZ = 5;  // MPI process split
    int OMPX = 1, OMPY = 1, OMPZ = 1;  // OpenMP thread split

    // Parse command line arguments for MPI and OpenMP splits
    if (argc > 6) 
    {
        MPIX = atoi(argv[1]);
        MPIY = atoi(argv[2]);
        MPIZ = atoi(argv[3]);
        OMPX = atoi(argv[4]);
        OMPY = atoi(argv[5]);
        OMPZ = atoi(argv[6]);
    }

    // Set number of OpenMP threads
    int total_threads = OMPX * OMPY * OMPZ;
    omp_set_num_threads(total_threads);

    // Calculate local grid dimensions
    int nx = GRID_SIZE / (MPIX * OMPX);
    int ny = GRID_SIZE / (MPIY * OMPY);
    int nz = GRID_SIZE / (MPIZ * OMPZ);

    if (rank == 0) 
    {
        printf("Configuration:\n");
        printf("MPI splits (X,Y,Z): %d,%d,%d\n", MPIX, MPIY, MPIZ);
        printf("OMP splits (X,Y,Z): %d,%d,%d\n", OMPX, OMPY, OMPZ);
        printf("Threads per MPI process: %d\n", total_threads);
    }

    // Set up cartesian topology
    int dims[3] = {MPIX, MPIY, MPIZ};
    int periods[3] = {0, 0, 0};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &cart_comm);
    
    int coords[3];
    MPI_Cart_coords(cart_comm, rank, 3, coords);

    // Initialize heat equation parameters
    double dx = PI / GRID_SIZE;
    double dy = PI / GRID_SIZE;
    double dz = PI / GRID_SIZE;
    double dx2 = dx * dx;
    double dy2 = dy * dy;
    double dz2 = dz * dz;
    double CFL = 0.25;
    double dt = CFL * dx2 / 3.0;
    int nsteps = (int)ceil(TIME_FINAL / dt);
    dt = TIME_FINAL / nsteps;

    // Allocate memory
    double *u = (double *)malloc((nx + 2) * (ny + 2) * (nz + 2) * sizeof(double));
    double *u_new = (double *)malloc((nx + 2) * (ny + 2) * (nz + 2) * sizeof(double));

    // Calculate offsets
    int x_offset = (coords[0] * nx * OMPX);
    int y_offset = (coords[1] * ny * OMPY);
    int z_offset = (coords[2] * nz * OMPZ);

    // Initialize grid using OpenMP
    #pragma omp parallel for collapse(3)
    for (int i = 1; i <= nx; i++) 
    {
        for (int j = 1; j <= ny; j++) 
        {
            for (int k = 1; k <= nz; k++) 
            {
                double x = (i - 1 + x_offset) * dx;
                double y = (j - 1 + y_offset) * dy;
                double z = (k - 1 + z_offset) * dz;
                u[I3DMPI(i, j, k, nx, ny)] = sin(x) * sin(y) * sin(z);
            }
        }
    }

    // Get neighboring processes
    int neighbors[6];
    MPI_Cart_shift(cart_comm, 2, 1, &neighbors[0], &neighbors[1]);
    MPI_Cart_shift(cart_comm, 1, 1, &neighbors[2], &neighbors[3]);
    MPI_Cart_shift(cart_comm, 0, 1, &neighbors[4], &neighbors[5]);

    double start_time = MPI_Wtime();

    // Main time stepping loop
    for (int t = 0; t < nsteps; t++) 
    {
        // Handle MPI communication first
        // Z TOP & BOTTOM
    if (neighbors[0] != MPI_PROC_NULL || neighbors[1] != MPI_PROC_NULL) 
    {
        MPI_Request requests[4];
        MPI_Status statuses[4];
        int req_count = 0;

        // RECIEVES
        if (neighbors[0] != MPI_PROC_NULL) { MPI_Irecv(&u[I3DMPI(1, 1, 0, nx, ny)], nx * ny, MPI_DOUBLE, neighbors[0], 1, cart_comm, &requests[req_count++]);}
        if (neighbors[1] != MPI_PROC_NULL) { MPI_Irecv(&u[I3DMPI(1, 1, nz + 1, nx, ny)], nx * ny, MPI_DOUBLE, neighbors[1], 0, cart_comm, &requests[req_count++]);}

        // SENDS
        if (neighbors[0] != MPI_PROC_NULL) { MPI_Isend(&u[I3DMPI(1, 1, 1, nx, ny)], nx * ny, MPI_DOUBLE, neighbors[0], 0, cart_comm, &requests[req_count++]); }
        if (neighbors[1] != MPI_PROC_NULL) { MPI_Isend(&u[I3DMPI(1, 1, nz, nx, ny)], nx * ny, MPI_DOUBLE, neighbors[1], 1, cart_comm, &requests[req_count++]); }

        MPI_Waitall(req_count, requests, statuses);
    }

    // Y LEFT & RIGHT
    if (neighbors[2] != MPI_PROC_NULL || neighbors[3] != MPI_PROC_NULL) 
    {
        MPI_Request requests[4];
        MPI_Status statuses[4];
        int req_count = 0;

        // RECIEVES
        if (neighbors[2] != MPI_PROC_NULL) { MPI_Irecv(&u[I3DMPI(1, 0, 1, nx, ny)], nx * nz, MPI_DOUBLE, neighbors[2], 3, cart_comm, &requests[req_count++]);}
        if (neighbors[3] != MPI_PROC_NULL) { MPI_Irecv(&u[I3DMPI(1, ny + 1, 1, nx, ny)], nx * nz, MPI_DOUBLE, neighbors[3], 2, cart_comm, &requests[req_count++]);}

        // SENDS
        if (neighbors[2] != MPI_PROC_NULL) {MPI_Isend(&u[I3DMPI(1, 1, 1, nx, ny)], nx * nz, MPI_DOUBLE, neighbors[2], 2, cart_comm, &requests[req_count++]);}
        if (neighbors[3] != MPI_PROC_NULL) {MPI_Isend(&u[I3DMPI(1, ny, 1, nx, ny)], nx * nz, MPI_DOUBLE, neighbors[3], 3, cart_comm, &requests[req_count++]);}

        MPI_Waitall(req_count, requests, statuses);
    }

    // X FRONT & BACK
    if (neighbors[4] != MPI_PROC_NULL || neighbors[5] != MPI_PROC_NULL) 
    {
        MPI_Request requests[4];
        MPI_Status statuses[4];
        int req_count = 0;

        // RECEIVES
        if (neighbors[4] != MPI_PROC_NULL) { MPI_Irecv(&u[I3DMPI(0, 1, 1, nx, ny)], ny * nz, MPI_DOUBLE, neighbors[4], 5, cart_comm, &requests[req_count++]);}
        if (neighbors[5] != MPI_PROC_NULL) { MPI_Irecv(&u[I3DMPI(nx + 1, 1, 1, nx, ny)], ny * nz, MPI_DOUBLE, neighbors[5], 4, cart_comm, &requests[req_count++]);}

        // SENDS
        if (neighbors[4] != MPI_PROC_NULL) { MPI_Isend(&u[I3DMPI(1, 1, 1, nx, ny)], ny * nz, MPI_DOUBLE, neighbors[4], 4, cart_comm, &requests[req_count++]);}
        if (neighbors[5] != MPI_PROC_NULL) { MPI_Isend(&u[I3DMPI(nx, 1, 1, nx, ny)], ny * nz, MPI_DOUBLE, neighbors[5], 5, cart_comm, &requests[req_count++]);}

        MPI_Waitall(req_count, requests, statuses);
    }
        // Update grid using OpenMP
        #pragma omp parallel for collapse(3)
        for (int i = 1; i <= nx; i++) 
        {
            for (int j = 1; j <= ny; j++) 
            {
                for (int k = 1; k <= nz; k++) 
                {
                    u_new[I3DMPI(i, j, k, nx, ny)] = u[I3DMPI(i, j, k, nx, ny)]
                        + dt / dx2 * (u[I3DMPI(i + 1, j, k, nx, ny)] - 2 * u[I3DMPI(i, j, k, nx, ny)] 
                        + u[I3DMPI(i - 1, j, k, nx, ny)])
                        + dt / dy2 * (u[I3DMPI(i, j + 1, k, nx, ny)] - 2 * u[I3DMPI(i, j, k, nx, ny)] 
                        + u[I3DMPI(i, j - 1, k, nx, ny)])
                        + dt / dz2 * (u[I3DMPI(i, j, k + 1, nx, ny)] - 2 * u[I3DMPI(i, j, k, nx, ny)] 
                        + u[I3DMPI(i, j, k - 1, nx, ny)]);
                }
            }
        }

        // Swap pointers
        double *tmp = u;
        u = u_new;
        u_new = tmp;
    }

    double end_time = MPI_Wtime();

    // Compute error using OpenMP
    double local_error = 0.0;
    #pragma omp parallel for collapse(3) reduction(+:local_error)
    for (int i = 1; i <= nx; i++) 
    {
        for (int j = 1; j <= ny; j++) 
        {
            for (int k = 1; k <= nz; k++) 
            {
                double x = (i - 1 + x_offset) * dx;
                double y = (j - 1 + y_offset) * dy;
                double z = (k - 1 + z_offset) * dz;
                double exact = exp(-3 * TIME_FINAL) * sin(x) * sin(y) * sin(z);
                double error = u[I3DMPI(i, j, k, nx, ny)] - exact;
                local_error += error * error * dx * dy * dz;
            }
        }
    }

    // Reduce error across all processes
    double global_error_squared;
    MPI_Reduce(&local_error, &global_error_squared, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) 
    {
        double global_error = sqrt(global_error_squared);
        printf("Global Error: %lf\n", global_error);
        printf("Runtime: %f seconds\n", end_time - start_time);
    }

    // Print node and thread binding information
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    
    #pragma omp parallel
    {
        #pragma omp critical
        {
            printf("Rank %d on node %s: Thread %d/%d\n", rank, processor_name, omp_get_thread_num(), omp_get_num_threads());
        }
    }

    free(u);
    free(u_new);
    MPI_Finalize();
    return 0;
}