data = read.csv("analyze_M_QC_lmer_effects.R.txt", sep="\t")


library(ggplot2)
plot = ggplot(data, aes(x=deletion_rate, y=-predictability_weight, fill=beta_TwoThree)) + geom_tile() + theme_bw() + scale_fill_gradient2() + xlab("delta") + ylab("lambda") + guides(fill=guide_legend(title="Effect")) + scale_y_reverse ()
ggsave("figures/effect-HasRC.pdf", width=4, height=3)
plot = ggplot(data, aes(x=deletion_rate, y=-predictability_weight, fill=-beta_EmbRate)) + geom_tile() + theme_bw() + scale_fill_gradient2() + xlab("delta") + ylab("lambda") + guides(fill=guide_legend(title="Effect"))
ggsave("figures/effect-EmbeddingRate.pdf", width=4, height=3)
plot = ggplot(data, aes(x=deletion_rate, y=-predictability_weight, fill=beta_Comp)) + geom_tile() + theme_bw() + scale_fill_gradient2() + xlab("delta") + ylab("lambda") + guides(fill=guide_legend(title="Effect"))
ggsave("figures/effect-Compatibility.pdf", width=4, height=3)
plot = ggplot(data, aes(x=deletion_rate, y=-predictability_weight, fill=beta_TwoThree.comp)) + geom_tile() + theme_bw() + scale_fill_gradient2() + xlab("delta") + ylab("lambda") + guides(fill=guide_legend(title="Effect"))
ggsave("figures/effect-HasRC-Compatibility.pdf", width=4, height=3)
plot = ggplot(data, aes(x=deletion_rate, y=-predictability_weight, fill=beta_TwoThree.EmbRate)) + geom_tile() + theme_bw() + scale_fill_gradient2() + xlab("delta") + ylab("lambda") + guides(fill=guide_legend(title="Effect"))
ggsave("figures/effect-HasRC-EmbRate.pdf", width=4, height=3)
plot = ggplot(data, aes(x=deletion_rate, y=-predictability_weight, fill=beta_comp.EmbRate)) + geom_tile() + theme_bw() + scale_fill_gradient2() + xlab("delta") + ylab("lambda") + guides(fill=guide_legend(title="Effect"))
ggsave("figures/effect-Compatibility-EmbRate.pdf", width=4, height=3)








