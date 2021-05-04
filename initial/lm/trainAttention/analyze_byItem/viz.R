data = read.csv("coefficients.tsv", sep="\t")


library(ggplot2)
plot = ggplot(data, aes(x=del, y=pred, fill=HasRC)) + geom_tile() + theme_bw() + scale_fill_gradient2()
plot = ggplot(data, aes(x=del, y=pred, fill=-Expect)) + geom_tile() + theme_bw() + scale_fill_gradient2()
plot = ggplot(data, aes(x=del, y=pred, fill=Compat)) + geom_tile() + theme_bw() + scale_fill_gradient2()





