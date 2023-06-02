library(parallel)

sa_parallel <- function(nHubs) {
    library(spsann)
    library(dplyr)
    library(reticulate)

    # ---------- import costEffectiveness.py ----------
    use_python('/apps/arch/2022r2/software/linux-rhel8-skylake_avx512/gcc-8.5.0/python-3.8.12-p6aunbmaoqlflowbsjqkzzm7n62qyrch/bin/python')
    source_python('costEffectiveness_v3.py')
    testFunction <- function(points, reverse) {
      testValue <- calcTotCostEffectiveness_r(points, reverse)
      return(testValue)
    }
    
    # ---------- Read data ----------
    candi <- read.csv('data/candiHubs_ams.csv')
    candi <- candi[,c("x", "y")]

    initialTemps <- read.csv('data/initialTemps.csv', sep = ";")
    initialTemps <- initialTemps[1:2]
    colnames(initialTemps) <- c('nHubs', 'trendline')
    
    # ---------- Adjust nHubs (initial temp, reverse) ----------
    if (nHubs < 80) {
      nHubs_adjusted = nHubs # no adjustment because nHubs is low 
      reverse <- FALSE
    } else {
      nHubs_adjusted = 139 - nHubs 
      reverse <- TRUE
    }
    
    # ---------- execute the simulated annealing algorithm ----------
    startTime <- Sys.time()
    
    # determine initial temperature  
    tempRow <- initialTemps[initialTemps$nHubs == nHubs, ]
    initialTemp <- tempRow$trendline 
    
    # calculate x.max and y.max 
    calcParams <- function(candi) {
      xCoords <- sort(candi$x)
      xDists <- sort(diff(xCoords))
      xMin <- xDists[1]
      xMax <- tail(xCoords, 1) - xCoords[1]
      yCoords <- sort(candi$y)
      yDists <- sort(diff(yCoords))
      yMin <- yDists[1]
      yMax <- tail(yCoords, 1) - yCoords[1]
      return(c(xMin, xMax, yMin, yMax))
    }
    paramsList <- calcParams(candi)
    
    # make schedule 
    schedule <- scheduleSPSANN(initial.temperature = initialTemp, cellsize = 0,
                               x.max = paramsList[2], y.max = paramsList[4])
    
    # run ssaa
    res <- optimUSER(
      points = nHubs_adjusted, fun = testFunction, reverse = reverse, 
      schedule = schedule, candi = candi, plotit = FALSE, track = TRUE
    ) 

    endTime <- Sys.time()

    # ---------- save results ----------
    resPoints <- res$points
    resPoints$nHubs <- nHubs
    resPoints$startTime <- startTime
    resPoints$endTime <- endTime 
    resPoints <- resPoints[, c('id', 'x', 'y', 'nHubs', 'startTime', 'endTime')]
    write.table(
    resPoints, file='results/resPoints_230502.csv', append=TRUE, sep=',', 
    col.names = FALSE
    )

    resEnergy <- res$objective$energy
    resEnergy$nHubs <- nHubs
    resEnergy$startTime <- startTime
    resEnergy$endTime <- endTime 
    write.table(
    resEnergy, file='results/resEnergy_230502.csv', append=TRUE, sep=',', 
    col.names = FALSE
    )

}

nCores <- detectCores()
cl <- makeCluster(nCores)
parSapply(cl, 2:135, sa_parallel)
stopCluster(cl)