=========================
	panda-yarn
	lihui@indiana.edu
	7/4/2013
=========================

panda version 0.43 can work on yarn with the help of mpich2-yarn
mpich2-yarn is referenced to clarkyzl's work at https://github.com/clarkyzl/mpich2-yarn
 
== installation and deployment panda-yarn ==

1) steps to compile and deploy mpich2-yarn  
   README.md at https://github.com/clarkyzl/mpich2-yarn

2) compile and panda code on gpu cluster

    modify the include and lib pathes of mpi and cuda in makefile, respectively.
    make -f Makefile
    mpiexec -l -machinefile nodes -n 2 ./panda_cmeans [numEvents][numDims]

3) run panda-yarn code

    hadoop jar mpich2-yarn-1.0-SNAPSHOT.jar -a panda_cmeans [numEvents][numDims]
    