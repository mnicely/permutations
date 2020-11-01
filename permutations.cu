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
#include <cassert>
#include <chrono>
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

template<typename T, typename U, uint TPB, uint P, size_t LAST_BLOCK>
__global__ void __launch_bounds__( TPB ) permute_shared( const U n, const U r_n, T *output ) {

    const auto block { cg::this_thread_block( ) };
    const auto tile32 { cg::tiled_partition<ws>( block ) };

    U tid { blockIdx.x * blockDim.x + threadIdx.x };
    U stride { blockDim.x * gridDim.x };
    U block_id { blockIdx.x };

    __shared__ unsigned char s[P * TPB];

    for ( U gid = tid; gid < r_n; gid += stride ) {

        // Reset shared memory
        for ( U i = 0; i < P; i++ ) {
            s[i * TPB + threadIdx.x] = 0;
        }
        block.sync( );

        // Compute factoradic
        if ( gid < n ) {

            U quo { gid };
            s[block.thread_rank( ) * P + ( P - 1 )] = 0;

#pragma unroll
            for ( int i = 2; i <= P; i++ ) {
                s[block.thread_rank( ) * P + ( P - i )] = quo % i;
                quo /= i;
            }
        }
        block.sync( );

        // Compute lehmer code : Each set gets a single warp
        for ( int w = tile32.meta_group_rank( ); w < TPB; w += tile32.meta_group_size( ) ) {

            uint key { tile32.thread_rank( ) };

#pragma unroll
            for ( int p = 0; p < P; p++ ) {
                uint rem { s[w * P + p] };

                __syncwarp( );

                if ( tile32.thread_rank( ) == rem ) {
                    s[w * P + p] = key;
                }

                if ( tile32.thread_rank( ) >= rem ) {
                    auto active = cg::coalesced_threads( );
                    key         = active.shfl_down( key, 1 );
                    active.sync( );
                }
            }
        }
        block.sync( );

        // if ( gid == ( n - 10 ) ) {
        //     for ( int g = 0; g < 10; g++ ) {
        //         printf( "%d: %d ", gid + g, ( block.thread_rank( ) + g ) );
        //         for ( int p = 0; p < P; p++ ) {
        //             printf( "%d", s[( block.thread_rank( ) + g ) * P + p] );
        //         }
        //         printf( "\n" );
        //     }
        // }

        if ( block_id != LAST_BLOCK ) {
#pragma unroll
            for ( int p = 0; p < P; p++ ) {
                output[( block_id * P * TPB ) + ( p * TPB ) + block.thread_rank( )] = s[p * TPB + block.thread_rank( )];
            }
        } else {
            uint block_size { TPB - ( r_n - n ) };
            // if ( block.thread_rank( ) == 0 )
            //     printf( "block_size %d\n", TPB - ( r_n - n ) );

            // if ( block.thread_rank( ) < block_size )
            //     printf( "%d: %d%d%d%d%d%d\n",
            //             block.thread_rank( ),
            //             s[block.thread_rank( ) * P + 0],
            //             s[block.thread_rank( ) * P + 1],
            //             s[block.thread_rank( ) * P + 2],
            //             s[block.thread_rank( ) * P + 3],
            //             s[block.thread_rank( ) * P + 4],
            //             s[block.thread_rank( ) * P + 5] );

            if ( block.thread_rank( ) < block_size ) {
#pragma unroll
                for ( int p = 0; p < P; p++ ) {
                    output[( block_id * P * TPB ) + ( p * block_size ) + block.thread_rank( )] =
                        s[p * block_size + block.thread_rank( )];
                }
            }
        }

        // if ( gid > ( n - 10 ) && gid < n )
        // printf( "L %d: %d%d%d%d\n",
        //         gid,
        //         s[block.thread_rank( ) * ws + 3],
        //         s[block.thread_rank( ) * ws + 2],
        //         s[block.thread_rank( ) * ws + 1],
        //         s[block.thread_rank( ) * ws + 0] );

        // Store

        block_id += gridDim.x;
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
template<typename T, uint P>
void findPermutations( const int &n, std::vector<T> &a, std::vector<T> &key ) {
    // Sort the given array
    std::sort( a.begin( ), a.end( ) );

    // Find all possible permutations
    size_t count {};
    do {
        for ( int i = 0; i < a.size( ); i++ ) {
            key[count * P + i] = a[i];
        }
        count++;
    } while ( next_permutation( a.begin( ), a.end( ) ) );
}

template<typename T, uint P>
void verify( const int &n, std::vector<T> &a, const T *data ) {
    // Sort the given array
    std::sort( a.begin( ), a.end( ) );

    // Find all possible permutations
    size_t count {};
    do {
        for ( int i = 0; i < a.size( ); i++ ) {
            if ( data[count * P + i] != a[i] ) {
                printf( "%lu: %d: %d %d\n", count, i, data[count * P + i], a[i] );
            }
        }
        count++;
    } while ( next_permutation( a.begin( ), a.end( ) ) );
}

/* Main */
int main( int argc, char **argv ) {

    using dtype = unsigned char;

    const uint   P { 10 };               // Cap at 32
    const size_t N { factorial( P ) };  // Number of sets, each set P values

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

    std::vector<dtype> h_key( N * P );

    std::vector<dtype> h_seq( P );
    std::iota( std::begin( h_seq ), std::end( h_seq ), 0 );

    std::printf( "\nSeq\n" );
    for ( auto &i : h_seq )
        std::printf( "%d ", i );
    std::printf( "\n" );

    // printf( "\nCPU\n" );
    // auto start = std::chrono::high_resolution_clock::now( );

    // findPermutations<dtype, P>( N, h_seq, h_key );

    // auto                                      stop           = std::chrono::high_resolution_clock::now( );
    // std::chrono::duration<double, std::milli> elapsed_cpu_ms = stop - start;
    // std::printf( "%0.2f ms\n", elapsed_cpu_ms.count( ) );

    // for ( int i = 0; i < N; i++ ) {
    //     for ( int j = 0; j < P; j++ ) {
    //         std::printf( "%d", h_key[i * P + j] );
    //     }
    //     std::printf( "\n" );
    // }

    // std::printf( "\n" );

    std::vector<dtype>    h_data( N * P );
    UniquePagedPtr<dtype> d_data { PagedAllocate<dtype>( N * P ) };

    const size_t num_blocks { static_cast<size_t>( N / tpb ) };
    const size_t pad_N { ( num_blocks + 1 ) * tpb };
    printf( "r_N = %lu: %lu\n", pad_N, num_blocks );

    void *args[] { const_cast<size_t *>( &N ), const_cast<size_t *>( &pad_N ), &d_data };

    for ( int i = 0; i < 1; i++ ) {
        if ( P < 13 ) {
            CUDA_RT_CALL(
                cudaLaunchKernel( reinterpret_cast<void *>( &permute_shared<dtype, uint, tpb, P, num_blocks> ),
                                  blocks_per_grid,
                                  threads_per_block,
                                  args,
                                  0,
                                  cuda_stream ) );
        }
    }

    CUDA_RT_CALL( cudaMemcpyAsync(
        h_data.data( ), d_data.get( ), N * P * sizeof( dtype ), cudaMemcpyDeviceToHost, cuda_stream ) );

    CUDA_RT_CALL( cudaDeviceSynchronize( ) );

    // std::printf( "\nGPU\n" );
    // for ( int i = 0; i < N; i++ ) {
    //     for ( int j = 0; j < P; j++ ) {
    //         std::printf( "%d", h_data[i * P + j] );
    //     }
    //     std::printf( "\n" );
    // }

    verify<dtype, P>(N, h_seq, h_data.data( ));

    CUDA_RT_CALL( cudaStreamDestroy( cuda_stream ) );

    return ( EXIT_SUCCESS );
}