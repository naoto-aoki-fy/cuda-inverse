#!/bin/sh

set -xe

# 1) コンパイル（ホスト／デバイスのオブジェクトを出力）
nvcc -dlto -arch=sm_90 -std=c++17 \
  -I./nvidia-mathdx-25.06.1/nvidia/mathdx/25.06/include/ \
  -I./cutlass-4.1.0/include \
  -c main.cu -o main.o

# 2) device-link：fatbin をここで渡す（fatbin は device-link でのみ許可）
nvcc --device-link -arch=sm_90 -dlto main.o \
  ./nvidia-mathdx-25.06.1/nvidia/mathdx/25.06/lib/libcusolverdx.fatbin \
  -o dlink.o

# 3) 最終リンク（ホスト側ライブラリを指定）
nvcc -arch=sm_90 main.o dlink.o -lcublas -o myprog
