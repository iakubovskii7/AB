library(BEST)
library(bayesAB)
library(abtest)

data = read.csv("/home/igor/Appbooster/proba.ai/AB/Description algorithms/Plots/bayes_ab.csv")

AB1 <- bayesTest(data$y1,
                 data$y2,
                 priors = c('alpha' = 1, 'beta' = 1),
                 n_samples = 1e6,
                 distribution = 'bernoulli')
summary(AB1)
plot(AB1)
0.02 / 0.0099

