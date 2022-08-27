data = read.csv("analyze_M_VN2_lmer_effects.R.txt", sep="\t")


library(ggplot2)
plot = ggplot(data, aes(x=deletion_rate, y=predictability_weight, fill=beta_TwoThree)) + geom_tile() + theme_bw() + scale_fill_gradient2() + xlab("delta") + ylab("lambda") + guides(fill=guide_legend(title="Effect")) + scale_y_reverse ()
ggsave("figures/effect-HasRC_VN2.pdf", width=4, height=3)
plot = ggplot(data, aes(x=deletion_rate, y=predictability_weight, fill=-beta_EmbRate)) + geom_tile() + theme_bw() + scale_fill_gradient2() + xlab("delta") + ylab("lambda") + guides(fill=guide_legend(title="Effect")) + scale_y_reverse ()
ggsave("figures/effect-EmbeddingRate_VN2.pdf", width=4, height=3)
plot = ggplot(data, aes(x=deletion_rate, y=predictability_weight, fill=beta_Comp)) + geom_tile() + theme_bw() + scale_fill_gradient2() + xlab("delta") + ylab("lambda") + guides(fill=guide_legend(title="Effect")) + scale_y_reverse ()
ggsave("figures/effect-Compatibility_VN2.pdf", width=4, height=3)
plot = ggplot(data, aes(x=deletion_rate, y=predictability_weight, fill=beta_TwoThree.comp)) + geom_tile() + theme_bw() + scale_fill_gradient2() + xlab("delta") + ylab("lambda") + guides(fill=guide_legend(title="Effect")) + scale_y_reverse ()
ggsave("figures/effect-HasRC-Compatibility_VN2.pdf", width=4, height=3)
plot = ggplot(data, aes(x=deletion_rate, y=predictability_weight, fill=beta_TwoThree.EmbRate)) + geom_tile() + theme_bw() + scale_fill_gradient2() + xlab("delta") + ylab("lambda") + guides(fill=guide_legend(title="Effect")) + scale_y_reverse ()
ggsave("figures/effect-HasRC-EmbRate_VN2.pdf", width=4, height=3)
plot = ggplot(data, aes(x=deletion_rate, y=predictability_weight, fill=beta_comp.EmbRate)) + geom_tile() + theme_bw() + scale_fill_gradient2() + xlab("delta") + ylab("lambda") + guides(fill=guide_legend(title="Effect")) + scale_y_reverse ()
ggsave("figures/effect-Compatibility-EmbRate_VN2.pdf", width=4, height=3)

plot = ggplot(data, aes(x=deletion_rate, y=predictability_weight, fill=beta_Comp + 0.5*beta_TwoThree.comp)) + geom_tile() + theme_bw() + scale_fill_gradient2() + xlab("delta") + ylab("lambda") + guides(fill=guide_legend(title="Effect")) + scale_y_reverse ()
plot = ggplot(data, aes(x=deletion_rate, y=predictability_weight, fill=beta_Comp - 0.5*beta_TwoThree.comp)) + geom_tile() + theme_bw() + scale_fill_gradient2() + xlab("delta") + ylab("lambda") + guides(fill=guide_legend(title="Effect")) + scale_y_reverse ()


data = read.csv("analyze_M_VN2_lmer_effects.R_tvalues.txt", sep="\t")


library(ggplot2)
plot = ggplot(data, aes(x=deletion_rate, y=predictability_weight, fill=beta_TwoThree)) + geom_tile() + theme_bw() + scale_fill_gradient2() + xlab("delta") + ylab("lambda") + guides(fill=guide_legend(title="Effect")) + scale_y_reverse ()
ggsave("figures/effect-HasRC_VN2_t.pdf", width=4, height=3)
plot = ggplot(data, aes(x=deletion_rate, y=predictability_weight, fill=-beta_EmbRate)) + geom_tile() + theme_bw() + scale_fill_gradient2() + xlab("delta") + ylab("lambda") + guides(fill=guide_legend(title="Effect")) + scale_y_reverse ()
ggsave("figures/effect-EmbeddingRate_VN2_t.pdf", width=4, height=3)
plot = ggplot(data, aes(x=deletion_rate, y=predictability_weight, fill=0.0+(abs(beta_Comp)>2))) + geom_tile() + theme_bw() + scale_fill_gradient2() + xlab("delta") + ylab("lambda") + guides(fill=guide_legend(title="Effect")) + scale_y_reverse ()
ggsave("figures/effect-Compatibility_VN2_t.pdf", width=4, height=3)
plot = ggplot(data, aes(x=deletion_rate, y=predictability_weight, fill=beta_TwoThree.comp)) + geom_tile() + theme_bw() + scale_fill_gradient2() + xlab("delta") + ylab("lambda") + guides(fill=guide_legend(title="Effect")) + scale_y_reverse ()
ggsave("figures/effect-HasRC-Compatibility_VN2_t.pdf", width=4, height=3)
plot = ggplot(data, aes(x=deletion_rate, y=predictability_weight, fill=beta_TwoThree.EmbRate)) + geom_tile() + theme_bw() + scale_fill_gradient2() + xlab("delta") + ylab("lambda") + guides(fill=guide_legend(title="Effect")) + scale_y_reverse ()
ggsave("figures/effect-HasRC-EmbRate_VN2_t.pdf", width=4, height=3)
plot = ggplot(data, aes(x=deletion_rate, y=predictability_weight, fill=beta_comp.EmbRate)) + geom_tile() + theme_bw() + scale_fill_gradient2() + xlab("delta") + ylab("lambda") + guides(fill=guide_legend(title="Effect")) + scale_y_reverse ()
ggsave("figures/effect-Compatibility-EmbRate_VN2_t.pdf", width=4, height=3)












