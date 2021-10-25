library(BEST)
y1 <- rbeta(1000, 100, 200)
y2 <- rbeta(900, 150, 250)
BESTout <- BESTmcmc(y1, y2, parallel=TRUE)

summary(BESTout)

BESTout$mu1 -
