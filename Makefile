CC ?= gcc
CFLAGS = -O3 -fPIC -std=c99 -Wall -fopenmp
LDFLAGS = -shared -fopenmp
SRCS = src/RKDP.c
OUTDIR = numba_RKDP/lib

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    OUT = $(OUTDIR)/libRKDP.dylib
else ifeq ($(OS),Windows_NT)
    OUT = $(OUTDIR)/libRKDP.dll
    LDFLAGS = -shared
else
    OUT = $(OUTDIR)/libRKDP.so
endif

.PHONY: all clean
all: $(OUT)

$(OUTDIR):
	mkdir -p $(OUTDIR)

$(OUT): $(SRCS) | $(OUTDIR)
	$(CC) $(CFLAGS) $(SRCS) -o $(OUT) $(LDFLAGS) -lm

clean:
	rm -rf $(OUTDIR)
