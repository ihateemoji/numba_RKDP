CC = gcc
CFLAGS = -O3 -fPIC -std=c99 -Wall -fopenmp
LDFLAGS = -shared -fopenmp
SRCS = src/RKDP.c
OUTDIR = numba_RKDP/lib
OUT = $(OUTDIR)/libRKDP.so

.PHONY: all clean
all: $(OUT)

$(OUTDIR):
	mkdir -p $(OUTDIR)

$(OUT): $(SRCS) | $(OUTDIR)
	$(CC) $(CFLAGS) $(SRCS) -o $(OUT) $(LDFLAGS) -lm

clean:
	rm -rf $(OUTDIR)
