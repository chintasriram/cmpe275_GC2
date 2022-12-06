// importing mpi.h and other libraries
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <limits.h>
#include <string.h>
#include <unistd.h>

#define MAX_SAMPLE_SIZE 128000000 // Defining default sample size
#define ROOT 0

// The belows are the modes we can use to run the code
enum run_modes
{
    REGULAR,
    TIMER,
    DEBUG
};

// Setting names for process Id's and default modes
int process_id;
int mode = REGULAR;
// Functions used in the project
void quicksort(int *array, int low, int high); // Implementing the quick sort
void swap(int *x, int *y);                     // Swapping the values
int subsort(int *array, int low, int high);    // subsort to implement quick sort
float getTimeLapse(struct timeval *last);      // Get the time

// Swap

void swap(int *x, int *y)
{
    int temp = *x; // temp holder
    *x = *y;       // assign x to y value
    *y = temp;     // assign y to held x value
}

// Quicksort routine
// Pass in array and lower and upper index to sort
// Calls subsort function to which helps to sort
// back the index of the final resting place of the pivot
// value.  Partially sorted upper and lower array is then
// recursively sorted.
void quicksort(int *array, int low, int high)
{
    if (high > low)
    {
        // final is the index where the pivot value is placed at its final destination
        int final = subsort(array, low, high);
        // Recurvisely call quicksort to fully sort the partially sorted section before and after the final
        quicksort(array, low, final - 1);
        quicksort(array, final + 1, high);
    }
}

// Subsort function Supports quicksort and actually does the sorting

int subsort(int *array, int low, int high)
{
    int pivot = array[high];
    int i = (low - 1); // Index of last element found to be <= pivot
    int j;             // Current iterator

#pragma omp parallel for
    for (j = low; j <= high - 1; j++)
    {
        // If current element is smaller than or equal to pivot
        if (array[j] <= pivot)
        {
#pragma omp critical
            // move to next insert location & swap to highest lower end
            {
                i++;
                swap(&array[i], &array[j]);
            }
        }
    }
    swap(&array[i + 1], &array[high]); // swap pivot to highest lower end
    return (i + 1);                    // send that location to quicksort
}

// Get time elapsed from last time recorded and it must be called on all processes due to MPI_Barrier
// Returns the elapsed time and refreshes the checkpoint
float getTimeLapse(struct timeval *last)
{
    struct timeval current; // current time
    float elapsed_time;     // elapsed time

    // Wait for all the processes
    MPI_Barrier(MPI_COMM_WORLD);

    // Fetch the present time
    gettimeofday(&current, NULL);
    elapsed_time = ((current.tv_sec - last->tv_sec) +
                    (current.tv_usec - last->tv_usec) / (float)1000000);

    // Updating the checkpoint value
    *last = current;
    // Return elapsed time
    return elapsed_time;
}

// Argument descriptions
//  argv[0] = program
//                        argv[1] = number of numbers_size
//                        argv[2] = run mode:
//                        REGULAR - default run mode
//                        DEBUG - print debug statements
//                        TIMED - print time statements (profiling)

int main(int argc, char **argv)
{
    // Checkpoints for program profiling and seed for srand()
    struct timeval start, stop, checkpoint, seed;
    float elapsed_time;
    int p_count; // Numbers of processors
    int i, j, offset, count;
    int flag = 1; // flag used to send messages
    // Local sample collection
    int *numbers;
    int numbers_size = MAX_SAMPLE_SIZE;
    // Regular samples sent/received from root
    int *samples;
    int sample_count, sample_size;
    // Total regular samples collected (only allocated by root)
    int *collective;
    int collective_size;
    // Final sorted sample collection
    int final_size;
    int *finalSorted;

    // Array for send , Recieve and Displacement
    int sDispl, rDispl;                             // count variables
    int *sendCount = malloc(p_count * sizeof(int)); // send count to each proc
    int *recvCount = malloc(p_count * sizeof(int)); // recv count from each proc
    int *sendDispl = malloc(p_count * sizeof(int)); // send displacement
    int *recvDispl = malloc(p_count * sizeof(int));

    // MPI Initializing
    MPI_Init(&argc, &argv);                     // Initialize MPI Environment
    MPI_Comm_size(MPI_COMM_WORLD, &p_count);    // Get the number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id); // Get the rank of the process

    // Seed the RNG using process id
    gettimeofday(&seed, NULL);
    srand((seed.tv_sec * 1000) + (seed.tv_usec / 1000) + process_id);

    // Print the run details and initialize run settings
    if (process_id == ROOT)
    {
        printf("Implementing PSRS Algo \n");
        printf(":::::::::::::::::::::::::::::::::::::::::::::::::::::::\n");
    }
    // Check program args
    switch (argc)
    {
    case 1: // REGULAR RUN
        if (process_id == ROOT)
        {
            printf("Normal run. Processors = %d,\n", p_count);
            printf("            Samples/Processor = %d\n", numbers_size);
        }
        break;
    case 3: // DEBUG or TIMER RUN
        if (strcmp(argv[2], "DEBUG") == 0)
        {
            if (process_id == ROOT)
                printf("DEBUG MODE\n");
            mode = DEBUG;
        }
        else if (strcmp(argv[2], "TIMER") == 0)
        {
            if (process_id == ROOT)
                printf("TIMER MODE\n");
            mode = TIMER;
        }
        else
        {
            if (process_id == ROOT)
            {
                printf("Invalid run args.  Second option must be \"DEBUG\" or \"TIMER\"\n");
                printf("Usage: psrs [<sample count> [ DEBUG | TIMER ]]\n");
            }
            MPI_Finalize();
            return 1;
        }
    case 2: // Custom sample size
        numbers_size = atoi(argv[1]);
        // Valid size
        if (numbers_size < 1)
        {
            if (process_id == ROOT)
            {
                printf("Invalid run args.  Sample size must be greater than 0.\n");
                printf("Usage: psrs [<sample count> [ DEBUG | TIMER]]\n");
                MPI_Finalize();
                return 1;
            }
        }
        // Print run values
        if (process_id == ROOT)
        {
            printf("Custom run\nProcessors = %d,\n", p_count);
            printf("Samples/Processor = %d\n", numbers_size);
        }
        break;
    default:
        if (process_id == ROOT)
        {
            printf("Error: Run Arguments are Invalid.\n");
            printf("Usage: psrs [<sample count> [ DEBUG | TIMER ]]\n");
        }
        MPI_Finalize();
        return 1;
    }
    if (process_id == ROOT)
        printf("::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n");

    // Allocate local collection
    numbers = malloc(numbers_size * sizeof(int));

    // Fill each local collection with random samples
    if (mode == DEBUG)
    { // For Debug mode we are making range 1 - 30
        for (i = 0; i < numbers_size; i++)
        {
            numbers[i] = rand() % 30;
        }
    }
    else
    { // for other, we are considering any 32 bit val
        for (i = 0; i < numbers_size; i++)
        {
            numbers[i] = rand();
        }
    }

    //// DEBUG PRINT for LOCAL COLLECTIONS
    if (mode == DEBUG)
    {
        // Printing every process collection
        if (process_id == ROOT)
        {
            printf("\nINITIAL  COLLECTIONS\n");
            printf("::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n");
        }
        if (process_id)
            MPI_Recv(&flag, 1, MPI_INT, process_id - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process %d: ", process_id);
        for (i = 0; i < numbers_size; i++)
        {
            printf("%d ", numbers[i]);
        }
        printf("\n");
        if (process_id < (p_count - 1))
            MPI_Send(&flag, 1, MPI_INT, process_id + 1, 1, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Start of Timer
    MPI_Barrier(MPI_COMM_WORLD);
    gettimeofday(&start, NULL);
    checkpoint = start;

    // Quicksort local collections
    quicksort(numbers, 0, numbers_size - 1);

    // Time Interval
    if (mode == TIMER)
    {
        elapsed_time = getTimeLapse(&checkpoint);
        if (process_id == ROOT)
            printf("Quicksort for  collection: \t%f\n", elapsed_time);
    }

    //// DEBUG PRINT - SORTED LOCAL COLLECTIONS
    if (mode == DEBUG)
    {
        // Print each process collection
        if (process_id == ROOT)
        {
            printf("\nSORTED  COLLECTIONS\n");
            printf("::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n");
        }
        if (process_id)
            MPI_Recv(&flag, 1, MPI_INT, process_id - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process %d: ", process_id);
        for (i = 0; i < numbers_size; i++)
        {
            printf("%d ", numbers[i]);
        }
        printf("\n");
        if (process_id < (p_count - 1))
            MPI_Send(&flag, 1, MPI_INT, process_id + 1, 1, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Regular Sampling size -- handle when p_count less than number of samples
    sample_count = (p_count < numbers_size) ? p_count : numbers_size;

    // Assigning space for regualr samples
    samples = malloc(sample_count * sizeof(int));
    offset = numbers_size / sample_count; // space out selections evenly
    j = offset - 1;                       // index of first one
    for (i = 0; i < sample_count; i++)
    {
        samples[i] = numbers[j];
        j += offset;
    }

    // Time Interval
    if (mode == TIMER)
    {
        elapsed_time = getTimeLapse(&checkpoint);
        if (process_id == ROOT)
            printf("Fetch regular samples: \t\t%f\n", elapsed_time);
    }

    //// DEBUG PRINT for REGULAR SAMPLES
    if (mode == DEBUG)
    {
        if (process_id == ROOT)
        {
            printf("\nREGULAR SAMPLES\n");
            printf(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n");
        }
        if (process_id)
            MPI_Recv(&flag, 1, MPI_INT, process_id - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // Samples Array
        printf("Process %d: ", process_id);
        for (i = 0; i < sample_count; i++)
        {
            printf("%d ", samples[i]);
        }

        printf("\n");
        if (process_id < (p_count - 1))
            MPI_Send(&flag, 1, MPI_INT, process_id + 1, 1, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Assigin space for the total collection of all regular samples
    collective_size = p_count * sample_count; // total size of array
    if (process_id == ROOT)                   // only root process needs to make this
        collective = malloc(collective_size * sizeof(int));

    // Gather all regular samples to root process
    MPI_Gather(samples, sample_count, MPI_INT, collective, sample_count, MPI_INT, ROOT, MPI_COMM_WORLD);

    // Free samples array and re-allocate space for split values
    free(samples);
    sample_size = p_count;          // size of array
    sample_count = sample_size - 1; // only p_count - 1 split values received; last one is MPI_MAX
    samples = malloc(sample_size * sizeof(int));

    // Time Interval
    if (mode == TIMER)
    {
        elapsed_time = getTimeLapse(&checkpoint);
        if (process_id == ROOT)
            printf("Gathering all regular samples from processes: \t%f\n", elapsed_time);
    }

    // Quicksort all regular samples
    // Then, pick out split values
    if (process_id == ROOT)
    {
        quicksort(collective, 0, collective_size - 1);

        // Create array of split values to broadcast back
        offset = collective_size / p_count; // pick values evenly spaced
        j = offset - 1;                     // index of first one
        for (i = 0; i < sample_count; i++)
        {
            samples[i] = collective[j];
            j += offset;
        }
        samples[sample_count] = INT_MAX - 1; // last section is everything less than INT_MAX
    }

    // Time Interval
    if (mode == TIMER)
    {
        elapsed_time = getTimeLapse(&checkpoint);
        if (process_id == ROOT)
            printf("Quicksort and select split values\n from all regular samples:\t\t%f\n", elapsed_time);
    }

    // Broadcast split values to all processes
    MPI_Bcast(samples, sample_count, MPI_INT, ROOT, MPI_COMM_WORLD);

    //// DEBUG PRINT - TOTAL REGULAR SAMPLES and BROADCASTED PIVOT VALUES
    if (mode == DEBUG)
    {
        if (process_id == ROOT)
        {
            printf("\nSORTED REGULAR SAMPLES(ALL)\n");
            printf("::::::::::::::::::::::::::::::::::::::::::::::::::\n");
            for (i = 0; i < collective_size; i++)
            {
                printf("%d ", collective[i]);
            }
            printf("\n");
        }
        MPI_Barrier(MPI_COMM_WORLD);

        if (process_id == 1)
        {
            printf("\nBROADCASTED SPLIT VALUES FROM ROOT\n");
            printf("::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n");
            printf("Process %d: ", process_id);
            for (i = 0; i < sample_count; i++)
            {
                printf("%d ", samples[i]);
            }
            printf("\n");
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Time Interval
    if (mode == TIMER)
    {
        elapsed_time = getTimeLapse(&checkpoint);
        if (process_id == ROOT)
            printf("Broadcast split values: \t%f\n", elapsed_time);
    }

    // Calculate the send count to each processor
    count = 0;
    j = 0; // local sorted array index iterator
    for (i = 0; i < p_count; i++)
    {
        while (numbers[j] <= samples[i] && j < numbers_size)
        {
            count++;
            j++;
        }
        sendCount[i] = count;
        count = 0;
    }

    // Time Interval
    if (mode == TIMER)
    {
        elapsed_time = getTimeLapse(&checkpoint);
        if (process_id == ROOT)
            printf("Calculate send counts: \t\t%f\n", elapsed_time);
    }

    // Send All-to-all exchange so all processes know what how much space
    // to allocate for final array
    // sendCount -> recvCount
    MPI_Alltoall(sendCount, 1, MPI_INT, recvCount, 1, MPI_INT, MPI_COMM_WORLD);

    // Time Interval
    if (mode == TIMER)
    {
        elapsed_time = getTimeLapse(&checkpoint);
        if (process_id == ROOT)
            printf("Send all-to-all: \t\t%f\n", elapsed_time);
    }

    // Calculate the sendDispl and recvDispl arrays
    sDispl = rDispl = 0;
    for (i = 0; i < p_count; i++)
    {
        sendDispl[i] = sDispl; // assign displacement
        recvDispl[i] = rDispl;
        sDispl += sendCount[i]; // add counts for next displacement
        rDispl += recvCount[i];
    }

    // Allocate space for final sorted array
    final_size = rDispl; // Last rDispl is the full array size
    finalSorted = malloc(final_size * sizeof(int));

    // Time Interval
    if (mode == TIMER)
    {
        elapsed_time = getTimeLapse(&checkpoint);
        if (process_id == ROOT)
            printf("Calc send/recv displ: \t\t%f\n", elapsed_time);
    }

    // Send final variable all-to-all message with the constructed arrays:
    // sendCount, sendDispl, recvCount, recvDispl
    MPI_Alltoallv(numbers, sendCount, sendDispl, MPI_INT,
                  finalSorted, recvCount, recvDispl, MPI_INT, MPI_COMM_WORLD);

    // Time Interval
    if (mode == TIMER)
    {
        elapsed_time = getTimeLapse(&checkpoint);
        if (process_id == ROOT)
            printf("Send all-to-allv: \t\t%f\n", elapsed_time);
    }

    // Quicksort the final arrays
    quicksort(finalSorted, 0, final_size - 1);

    // Time Interval
    if (mode == TIMER)
    {
        elapsed_time = getTimeLapse(&checkpoint);
        if (process_id == ROOT)
            printf("Final quicksort: \t\t%f\n", elapsed_time);
    }

    // BARRIER - Wait for all processes to finish before getting end time
    MPI_Barrier(MPI_COMM_WORLD);
    gettimeofday(&stop, NULL); // Record end time

    //// DEBUG PRINT - SEND COUNTS, RECEIVE COUNTS, FINAL SORTED
    if (mode == DEBUG)
    {
        if (process_id == ROOT)
        {
            printf("\nSEND COUNTS\n");
            printf("::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n");
        }
        // DEBUG Sorted Collections
        if (process_id)
            MPI_Recv(&flag, 1, MPI_INT, process_id - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        printf("Process %d: ", process_id);
        for (i = 0; i < p_count; i++)
        {
            printf("%d ", sendCount[i]);
        }
        printf("\n");
        if (process_id < (p_count - 1))
            MPI_Send(&flag, 1, MPI_INT, process_id + 1, 1, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        if (process_id)
            MPI_Recv(&flag, 1, MPI_INT, process_id - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (process_id == ROOT)
        {
            printf("\nRECEIVE COUNTS\n");
            printf("::::::::::::::::::::::::::::::::::::::::\n");
        }
        printf("Process %d: ", process_id);
        for (i = 0; i < p_count; i++)
        {
            printf("%d ", recvCount[i]);
        }
        printf("\n");
        if (process_id < (p_count - 1))
            MPI_Send(&flag, 1, MPI_INT, process_id + 1, 1, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        if (process_id)
            MPI_Recv(&flag, 1, MPI_INT, process_id - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (process_id == ROOT)
        {
            printf("\nFINAL SORTED\n");
            printf(":::::::::::::::::::::::::::::::::::::::::::::::::::::::\n");
        }
        printf("Process %d: ", process_id);
        for (i = 0; i < final_size; i++)
        {
            printf("%d ", finalSorted[i]);
        }
        printf("\n");

        sleep(1);
        if (process_id < (p_count - 1))
            MPI_Send(&flag, 1, MPI_INT, process_id + 1, 1, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // PRINT FINAL TIME
    checkpoint = start;
    elapsed_time = getTimeLapse(&checkpoint);
    if (process_id == ROOT)
        printf("Final time: \t\t\t%f\n", elapsed_time);

    // Free all allocated memory
    free(sendCount);
    free(recvCount);
    free(sendDispl);
    free(recvDispl);
    free(numbers);
    free(samples);
    free(collective);
    free(finalSorted);

    MPI_Finalize(); // Cleanup MPI
    return 0;       // exit
}
