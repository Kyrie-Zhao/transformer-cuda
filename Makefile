CUDA_INSTALL_PATH = /usr/local/cuda/
#/opt/cuda-9.1/
INCLUDES  := -I $(CUDA_INSTALL_PATH)/samples/common/inc -I $(CUDA_INSTALL_PATH)/include

CC				=	g++
NVCC		=	$(CUDA_INSTALL_PATH)/bin/nvcc -ccbin $(CC) --compiler-options -fno-strict-aliasing --std=c++11 -arch=sm_60  --ptxas-options=-v 
NVLIBS		=	-L $(CUDA_INSTALL_PATH)/lib  -L $(CUDA_INSTALL_PATH)/lib64  -lcudart -lcublas 



ARCHITECTURE	?=	unix

MPI				?=	no

HIGH_PRECISION 	?= 	no

CUDA			?=  yes

BITS_64         ?=  yes

GPROF			?=	no

DEBUG			?=	no

OPTFLAGS		?=	-O3

ifeq ($(strip $(ARCHITECTURE)), unix)
	USEREADLINE	?=	no
else
	USEREADLINE	?=	no
endif

ifeq ($(strip $(ARCHITECTURE)), unix)
	CFLAGS	+=	-DUNIX_VERSION
endif

ifeq ($(strip $(BITS_64)), yes)
	CFLAGS	+=	-D_64BIT
    CUFLAGS +=  -D_64BIT
endif

ifeq ($(strip $(HIGH_PRECISION)), yes)
     CFLAGS  +=  -DHIGH_PRECISION
     CUFLAGS +=  -DHIGH_PRECISION
 endif

ifeq ($(strip $(USEREADLINE)), yes)
	CFLAGS	+=	-DUSE_READLINE
	LIBS	+=	-lncurses -lreadline
endif

ifeq ($(strip $(CUDA)), yes)
	CFLAGS	+=	-DCUDA
	#CFLAGS	+=	--std=c++11
	#LDFLAGS	+=	-fPIC
	LIBS	+=	$(NVLIBS)
	OBJECTS	+=	transformer.o
	CUFILES	+=	transformer.cu transformer.h 
	GPROF	=	no
endif

ifeq ($(strip $(MPI)), yes)
	CFLAGS	+=	-DMPI_ENABLED
	CUFLAGS	+=	-DMPI
	CC	=	mpicc
endif

ifeq ($(strip $(GPROF)), yes)
	CFLAGS	+=	-pg
	LIBS	+=	-pg
endif

ifeq ($(strip $(DEBUG)), yes)
	CFLAGS	+=	-ggdb
else
	CFLAGS	+=	$(OPTFLAGS)
	CUFLAGS	+=	$(OPTFLAGS)
endif

#CFLAGS	+=	-Wall

LIBS	+=	-lm -lstdc++

LDFLAGS	+=	$(CFLAGS)
LDLIBS	=	$(LIBS)

#OBJECTS	+= cuda-frontend.o lstmrnnlm.o

#rnncuda : $(OBJECTS)
#	$(NVCC) $(INCLUDES) $(CFLAGS) $(OBJECTS) $(LDLIBS) -o rnncuda

all: build

build: transformer 
	
ifeq ($(strip $(CUDA)), yes)
transformer.o : $(CUFILES)
	$(NVCC) $(INCLUDES) $(CUFLAGS) -c transformer.cu
endif

#cuda-frontend.o: cuda-frontend.cpp cuda-frontend.h cuda-backend.h
#	$(NVCC) $(INCLUDES) $(CFLAGS) -c cuda-frontend.cpp  
#lstmrnnlm.o: lstmrnnlm.cpp lstmrnnlm.h cuda-frontend.h cuda-backend.h MPIClass.h
#	$(NVCC) $(INCLUDES) $(CFLAGS) -c lstmrnnlm.cpp 
	
transformer : $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) $(LDLIBS) -o transformer

run: build
	 ./transformer

#main.o : main.cpp rnnlmlib.h cuda-frontend.h cuda-backend.h
#	$(NVCC) $(CFLAGS) -c main.cpp

#mb.o : mb.c mb.h globals.h
#	$(CC) $(CFLAGS) -c mb.c
#bayes.o : bayes.c mb.h globals.h bayes.h command.h mcmc.o
#	$(CC) $(CFLAGS) -c bayes.c
#command.o : command.c mb.h globals.h command.h bayes.h model.h mcmc.h plot.h sump.h sumt.h
#	$(CC) $(CFLAGS) -c command.c
#mbmath.o : mbmath.c mb.h globals.h mbmath.h bayes.h
#	$(CC) $(CFLAGS) -c mbmath.c
#mcmc.o : mcmc.c mb.h globals.h bayes.h mcmc.h model.h command.h mbmath.h sump.h sumt.h plot.h
#	$(CC) $(CFLAGS) -c mcmc.c
#model.o : model.c mb.h globals.h bayes.h model.h command.h
#	$(CC) $(CFLAGS) -c model.c
#plot.o : plot.c mb.h globals.h command.h bayes.h plot.h sump.h
#	$(CC) $(CFLAGS) -c plot.c
#sump.o : sump.c mb.h globals.h command.h bayes.h sump.h mcmc.h
#	$(CC) $(CFLAGS) -c sump.c
#sumt.o : sumt.c mb.h globals.h command.h bayes.h mbmath.h sumt.h mcmc.h
#	$(CC) $(CFLAGS) -c sumt.c

.PHONY : clean
clean :
	-rm -f mb $(OBJECTS) transformer
