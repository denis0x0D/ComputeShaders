// Tiled and coalesced version
__kernel void myGEMM2(const int M, const int N, const int K,
                      const __global float *A, const __global float *B,
                      __global float *C) {

  // Thread identifiers
  const int TS = 4;
  const int local_row = get_local_id(0);            // Local row ID (max: TS)
  const int local_col = get_local_id(1);            // Local col ID (max: TS)
  const int globalRow = TS * get_group_id(0) + local_row; // Row ID of C (0..M)
  const int globalCol = TS * get_group_id(1) + local_col; // Col ID of C (0..N)

  // Local memory to fit a tile of TS*TS elements of A and B
  __local float Asub[TS][TS];
  __local float Bsub[TS][TS];

  // Initialise the accumulation register
  float acc = 0.0f;

  // Loop over all tiles
  const int numTiles = K / TS;
  for (int t = 0; t < numTiles; t++) {

    // Load one tile of A and B into local memory
    const int tiledRow = get_local_size(0) * t + local_row;
    const int tiledCol = get_local_size(1) * t + local_col;

    Asub[local_col][local_row] =
        A[(get_local_size(0) * t + local_col) * M +
          (get_local_size(0) * get_group_id(0) + local_row)];

    Bsub[local_col][local_row] = B[globalCol * K + tiledRow];

    // Synchronise to make sure the tile is loaded
    barrier(CLK_LOCAL_MEM_FENCE);

    // Perform the computation for a single tile
    for (int k = 0; k < TS; k++) {
      acc += Asub[k][local_row] * Bsub[local_col][k];
    }

    // Synchronise before loading the next tile
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Store the final result in C
  C[globalCol * M + globalRow] = acc;
}
