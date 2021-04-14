# Copyright (C) 2021  Maryam Razmjouei
# http://github.com/maryam-razmjouei

# Set up R environment
library(Rmpi)
library(parallel)
library(snow)
library(e1071)
set.seed(123)
source('../msvmRFE.R')
input <- read.csv(file.choose(), header = TRUE)

start.time <- Sys.time()
start.time

# Take a look at the expected input structure

dim(input)

# Set up cross validation
nfold = 4
nrows = nrow(input)
folds = rep(1:nfold, len=nrows)[sample(nrows)]
folds
folds = lapply(1:nfold, function(x) which(folds == x))

#make a cluster
cl <- makeMPIcluster(mpi.universe.size())

ad= source('../msvmRFE.R')

# Perform feature ranking on all tasks
clusterExport(cl, list("input","svmRFE","getWeights","svm"))
results <-parLapply(cl,folds, svmRFE.wrap,input, k=10, halve.above=100)

results

# Obtain top features across ALL folds
top.features = WriteFeatures(results, input, save=F)
head(top.features)

# Estimate generalization error using a varying number of top features
clusterExport(cl, list("top.features","results", "tune","tune.control"))
featsweep = parLapply(cl,1:4, FeatSweep.wrap, results, input)
featsweep

stopCluster(cl)
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken

# Make plot
no.info = min(prop.table(table(input[,1])))
errors = sapply(featsweep, function(x) ifelse(is.null(x), NA, x$error))

dev.new(width=4, height=4, bg='red')
PlotErrors(errors, no.info=no.info)
dev.off()
plot(top.features)
mpi.exit()

