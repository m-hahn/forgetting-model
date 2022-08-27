data = read.csv("coefficients.tsv", sep="\t")


library(ggplot2)
plot = ggplot(data, aes(x=del, y=pred, fill=HasRC)) + geom_tile() + theme_bw() + scale_fill_gradient2() + xlab("delta") + ylab("lambda") + guides(fill=guide_legend(title="Effect"))
ggsave("figures/effect-HasRC.pdf", width=4, height=3)
plot = ggplot(data, aes(x=del, y=pred, fill=-Expect)) + geom_tile() + theme_bw() + scale_fill_gradient2() + xlab("delta") + ylab("lambda") + guides(fill=guide_legend(title="Effect"))
ggsave("figures/effect-EmbeddingRate.pdf", width=4, height=3)
plot = ggplot(data, aes(x=del, y=pred, fill=Compat)) + geom_tile() + theme_bw() + scale_fill_gradient2() + xlab("delta") + ylab("lambda") + guides(fill=guide_legend(title="Effect"))
ggsave("figures/effect-Compatibility.pdf", width=4, height=3)





