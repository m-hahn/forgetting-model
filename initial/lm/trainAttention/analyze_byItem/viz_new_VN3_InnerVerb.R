
library(dplyr)
library(tidyr)
data = read.csv("analyze_M_VN3_248_InnerVerb_lmer_effects.R.txt", sep="\t") %>% filter(deletion_rate < 2)


rates = c(-2, -1.4, -2, -1.4)
hasrc = c(0.5,0.5, -0.5, -0.5)

data2 = data.frame(rates=rates, hasrc=hasrc)
data2 = merge(data, data2)

plot = ggplot(data2, aes(x=rates, y=beta_TwoThree*hasrc + beta_EmbRate * rates + beta_TwoThree.EmbRate * rates * hasrc, group=hasrc, color=hasrc)) + geom_line() + facet_grid(deletion_rate~predictability_weight)

library(ggplot2)
plot = ggplot(data, aes(x=deletion_rate, y=predictability_weight, fill=beta_TwoThree)) + geom_tile() + theme_bw() + scale_fill_gradient2() + xlab("delta") + ylab("lambda") + guides(fill=guide_legend(title="Effect")) + scale_y_reverse ()
ggsave("figures/effect-HasRC_VN3_248_InnerVerb.pdf", width=4, height=3)
plot = ggplot(data, aes(x=deletion_rate, y=predictability_weight, fill=-beta_EmbRate)) + geom_tile() + theme_bw() + scale_fill_gradient2() + xlab("delta") + ylab("lambda") + guides(fill=guide_legend(title="Effect")) + scale_y_reverse ()
ggsave("figures/effect-EmbeddingRate_VN3_248_InnerVerb.pdf", width=4, height=3)
plot = ggplot(data, aes(x=deletion_rate, y=predictability_weight, fill=beta_Comp)) + geom_tile() + theme_bw() + scale_fill_gradient2() + xlab("delta") + ylab("lambda") + guides(fill=guide_legend(title="Effect")) + scale_y_reverse ()
ggsave("figures/effect-Compatibility_VN3_248_InnerVerb.pdf", width=4, height=3)
plot = ggplot(data, aes(x=deletion_rate, y=predictability_weight, fill=beta_TwoThree.Comp)) + geom_tile() + theme_bw() + scale_fill_gradient2() + xlab("delta") + ylab("lambda") + guides(fill=guide_legend(title="Effect")) + scale_y_reverse ()
ggsave("figures/effect-HasRC-Compatibility_VN3_248_InnerVerb.pdf", width=4, height=3)
plot = ggplot(data, aes(x=deletion_rate, y=predictability_weight, fill=beta_TwoThree.EmbRate)) + geom_tile() + theme_bw() + scale_fill_gradient2() + xlab("delta") + ylab("lambda") + guides(fill=guide_legend(title="Effect")) + scale_y_reverse ()
ggsave("figures/effect-HasRC-EmbRate_VN3_248_InnerVerb.pdf", width=4, height=3)
plot = ggplot(data, aes(x=deletion_rate, y=predictability_weight, fill=beta_Comp.EmbRate)) + geom_tile() + theme_bw() + scale_fill_gradient2() + xlab("delta") + ylab("lambda") + guides(fill=guide_legend(title="Effect")) + scale_y_reverse ()
ggsave("figures/effect-Compatibility-EmbRate_VN3_248_InnerVerb.pdf", width=4, height=3)

plot = ggplot(data, aes(x=deletion_rate, y=predictability_weight, fill=beta_Comp + 0.5*beta_TwoThree.Comp)) + geom_tile() + theme_bw() + scale_fill_gradient2() + xlab("delta") + ylab("lambda") + guides(fill=guide_legend(title="Effect")) + scale_y_reverse ()
plot = ggplot(data, aes(x=deletion_rate, y=predictability_weight, fill=beta_Comp - 0.5*beta_TwoThree.Comp)) + geom_tile() + theme_bw() + scale_fill_gradient2() + xlab("delta") + ylab("lambda") + guides(fill=guide_legend(title="Effect")) + scale_y_reverse ()


data = read.csv("analyze_M_VN3_248_InnerVerb_lmer_effects.R_tvalues.txt", sep="\t") %>% filter(deletion_rate < 2)


library(ggplot2)
plot = ggplot(data, aes(x=deletion_rate, y=predictability_weight, fill=beta_TwoThree)) + geom_tile() + theme_bw() + scale_fill_gradient2() + xlab("delta") + ylab("lambda") + guides(fill=guide_legend(title="Effect")) + scale_y_reverse ()
ggsave("figures/effect-HasRC_VN3_248_InnerVerb_t.pdf", width=4, height=3)
plot = ggplot(data, aes(x=deletion_rate, y=predictability_weight, fill=-beta_EmbRate)) + geom_tile() + theme_bw() + scale_fill_gradient2() + xlab("delta") + ylab("lambda") + guides(fill=guide_legend(title="Effect")) + scale_y_reverse ()
ggsave("figures/effect-EmbeddingRate_VN3_248_InnerVerb_t.pdf", width=4, height=3)
plot = ggplot(data, aes(x=deletion_rate, y=predictability_weight, fill=0.0+((beta_Comp)))) + geom_tile() + theme_bw() + scale_fill_gradient2() + xlab("delta") + ylab("lambda") + guides(fill=guide_legend(title="Effect")) + scale_y_reverse ()
ggsave("figures/effect-Compatibility_VN3_248_InnerVerb_t.pdf", width=4, height=3)
plot = ggplot(data, aes(x=deletion_rate, y=predictability_weight, fill=beta_TwoThree.Comp)) + geom_tile() + theme_bw() + scale_fill_gradient2() + xlab("delta") + ylab("lambda") + guides(fill=guide_legend(title="Effect")) + scale_y_reverse ()
ggsave("figures/effect-HasRC-Compatibility_VN3_248_InnerVerb_t.pdf", width=4, height=3)
plot = ggplot(data, aes(x=deletion_rate, y=predictability_weight, fill=beta_TwoThree.EmbRate)) + geom_tile() + theme_bw() + scale_fill_gradient2() + xlab("delta") + ylab("lambda") + guides(fill=guide_legend(title="Effect")) + scale_y_reverse ()
ggsave("figures/effect-HasRC-EmbRate_VN3_248_InnerVerb_t.pdf", width=4, height=3)
plot = ggplot(data, aes(x=deletion_rate, y=predictability_weight, fill=beta_Comp.EmbRate)) + geom_tile() + theme_bw() + scale_fill_gradient2() + xlab("delta") + ylab("lambda") + guides(fill=guide_legend(title="Effect")) + scale_y_reverse ()
ggsave("figures/effect-Compatibility-EmbRate_VN3_248_InnerVerb_t.pdf", width=4, height=3)












