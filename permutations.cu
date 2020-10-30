/*
 * Copyright 1993-2020 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

#include <algorithm>
#include <memory>
#include <numeric>
#include <vector>

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// *************** FOR ERROR CHECKING *******************
#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL( call )                                                                                           \
    {                                                                                                                  \
        auto status = static_cast<cudaError_t>( call );                                                                \
        if ( status != cudaSuccess )                                                                                   \
            std::fprintf( stderr,                                                                                      \
                          "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "                                   \
                          "with "                                                                                      \
                          "%s (%d).\n",                                                                                \
                          #call,                                                                                       \
                          __LINE__,                                                                                    \
                          __FILE__,                                                                                    \
                          cudaGetErrorString( status ),                                                                \
                          status );                                                                                    \
    }
#endif  // CUDA_RT_CALL
// *************** FOR ERROR CHECKING *******************

// ***************** FOR NVTX MARKERS *******************
#ifdef USE_NVTX
#include "nvtx3/nvToolsExt.h"

const uint32_t colors[]   = { 0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff };
const int      num_colors = sizeof( colors ) / sizeof( uint32_t );

#define PUSH_RANGE( name, cid )                                                                                        \
    {                                                                                                                  \
        int color_id                      = cid;                                                                       \
        color_id                          = color_id % num_colors;                                                     \
        nvtxEventAttributes_t eventAttrib = { 0 };                                                                     \
        eventAttrib.version               = NVTX_VERSION;                                                              \
        eventAttrib.size                  = NVTX_EVENT_ATTRIB_STRUCT_SIZE;                                             \
        eventAttrib.colorType             = NVTX_COLOR_ARGB;                                                           \
        eventAttrib.color                 = colors[color_id];                                                          \
        eventAttrib.messageType           = NVTX_MESSAGE_TYPE_ASCII;                                                   \
        eventAttrib.message.ascii         = name;                                                                      \
        nvtxRangePushEx( &eventAttrib );                                                                               \
    }
#define POP_RANGE( ) nvtxRangePop( );
#else
#define PUSH_RANGE( name, cid )
#define POP_RANGE( )
#endif
// ***************** FOR NVTX MARKERS *******************

constexpr size_t factorial( const size_t &n ) {
    return ( n <= 1 ) ? 1 : ( n * factorial( n - 1 ) );
}

constexpr int tpb { 256 };
constexpr int ws { 32 };

template<typename T, uint TPB, uint P>
__global__ void permute_cuda( const int n, const int r_n, T *__restrict__ output ) {

    const auto block { cg::this_thread_block( ) };
    const auto tile32 { cg::tiled_partition<ws>( block ) };

    unsigned int tid { blockIdx.x * blockDim.x + threadIdx.x };
    unsigned int stride { blockDim.x * gridDim.x };

    __shared__ uint s[ws * TPB];

    for ( int gid = tid; gid < r_n; gid += stride ) {

        // Reset shared memory
        for ( int i = 0; i < ws; i++ ) {
            s[i * TPB + threadIdx.x] = 0;
        }
        block.sync( );

        // Compute factoradic
        if ( gid < n ) {

            int quo { gid };
            s[tid * ws] = 0;

#pragma unroll P
            for ( int i = 2; i <= P; i++ ) {
                s[tid * ws + ( i - 1 )] = static_cast<uint>( fmodf( quo, i ) );
                quo                     = static_cast<int>( __fdividef( quo, i ) );
            }
        }
        tile32.sync( );

        if ( tid == 0 )
            printf( "\n" );

        // Compute lehmer code : Each set gets a single warp
        for ( int w = tile32.meta_group_rank( ); w < n; w += tile32.meta_group_size( ) ) {

            if ( tile32.thread_rank( ) == 0 )
                printf( "%d: %d\n", tile32.thread_rank( ), w );
            uint key { tile32.thread_rank( ) };

            for ( int p = P - 1; p >= 0; p-- ) {
                uint rem { s[w * ws + p] };
                uint delta { 1 };
                if ( tile32.thread_rank( ) == rem ) {
                    s[w * ws + p] = key;
                }

                if ( tile32.thread_rank( ) >= rem ) {
                    auto active = cg::coalesced_threads( );
                    key         = active.shfl_down( key, delta );
                    active.sync( );
                }
            }
        }
        block.sync( );
        if ( tid < n )
            printf( "L %d: %d%d%d%d%d%d%d%d\n",
					tid,
					s[tid * ws + 7],
					s[tid * ws + 6],
					s[tid * ws + 5],
                    s[tid * ws + 4],
                    s[tid * ws + 3],
                    s[tid * ws + 2],
                    s[tid * ws + 1],
                    s[tid * ws + 0] );

        // Store
    }
}

template<typename T>
T *PagedAllocate( const size_t &N ) {
    T *    ptr { nullptr };
    size_t bytes { N * sizeof( T ) };
    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &ptr ), bytes ) );
    return ( ptr );
}

template<typename T>
struct PagedMemoryDeleter {
    void operator( )( T *ptr ) {
        if ( ptr ) {
            CUDA_RT_CALL( cudaFree( ptr ) );
        }
    }
};

template<typename T>
using UniquePagedPtr = std::unique_ptr<T, PagedMemoryDeleter<T>>;

// Function to find the permutations
template<typename T>
void findPermutations( std::vector<T> a, int n ) {
    // Sort the given array
    std::sort( a.begin( ), a.end( ) );

    // Find all possible permutations
    do {
        for ( auto &i : a )
            std::printf( "%d", i );
        std::printf( "\n" );
    } while ( next_permutation( a.begin( ), a.end( ) ) );
}

/* Main */
int main( int argc, char **argv ) {

    using dtype = char;

    const uint P { 4 };               // Cap at 32
    size_t     N { factorial( P ) };  // Number of sets, each set P values

    printf( "N = %lu\n", N );

    int device {};
    int sm_count {};
    int threads_per_block { tpb };
    int blocks_per_grid {};

    cudaStream_t cuda_stream;
    CUDA_RT_CALL( cudaStreamCreate( &cuda_stream ) );

    CUDA_RT_CALL( cudaGetDevice( &device ) );
    CUDA_RT_CALL( cudaDeviceGetAttribute( &sm_count, cudaDevAttrMultiProcessorCount, device ) );

    blocks_per_grid = sm_count * 32;
    // blocks_per_grid = 1;

    // Determine how much shared memory is required
    size_t shared_size { P * tpb * sizeof( int ) };

    std::vector<dtype> h_seq( P );
    std::iota( std::begin( h_seq ), std::end( h_seq ), 0 );

    std::printf( "\nKey\n" );
    for ( auto &i : h_seq )
        std::printf( "%d ", i );
    std::printf( "\n" );

    printf( "\nCPU\n" );
    findPermutations( h_seq, P );
    printf( "\n" );

    // std::vector<dtype>    h_key( P );
    std::vector<dtype>    h_data( N * P );
    UniquePagedPtr<dtype> d_data { PagedAllocate<dtype>( N * P ) };

    int r_N { static_cast<int>( N / tpb ) + 1 * tpb };
    printf( "r_N = %d\n", r_N );

    void *args[] { &N, &r_N, &d_data };

    CUDA_RT_CALL( cudaLaunchKernel( reinterpret_cast<void *>( &permute_cuda<dtype, tpb, P> ),
                                    blocks_per_grid,
                                    threads_per_block,
                                    args,
                                    shared_size,
                                    cuda_stream ) );

    // CUDA_RT_CALL( cudaMemcpyAsync(
    //     h_data.data( ), d_data.get( ), N * P * sizeof( dtype ), cudaMemcpyDeviceToHost, cuda_stream ) );

    CUDA_RT_CALL( cudaDeviceSynchronize( ) );

    // std::printf( "\nGPU\n" );
    // for ( int i = 0; i < N; i++ ) {
    //     for ( int j = 0; j < P; j++ ) {
    //         std::printf( "%d", h_data[i * P + j] );
    //     }
    //     std::printf( "\n" );
    // }

    CUDA_RT_CALL( cudaStreamDestroy( cuda_stream ) );

    return ( EXIT_SUCCESS );
}