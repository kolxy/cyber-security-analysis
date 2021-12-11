RequiredPackages <- c("dplyr", "ggplot2")

for (pkg in RequiredPackages) {
  if (pkg %in% rownames(installed.packages()) == FALSE)
  {install.packages(pkg)}
  if (pkg %in% rownames(.packages()) == FALSE)
  {library(pkg, character.only = TRUE)}
}

setwd("C:/Users/waltz/Documents/cyber-security-analysis/output")

# define color blind palette

color_blind_palette <- c("#009E73", "#D55E00")

# read data

df <- read.csv("f1_scores.csv",
               header=TRUE,
               sep=",",
               stringsAsFactors = TRUE)

# convert accuracy to percentages

df <- df %>% mutate(ML.Classifier = ifelse(ML.Classifier == "K-Nearest Neighbors", "KNN",
                                           ifelse(ML.Classifier == "Logistic Regression", "Log. Reg",
                                                  ifelse(ML.Classifier == "Multilayer Perceptron", "MLP", "RF"))))

# plot binary results

caption <- "KNN = K-Nearest Neighbors, Log. Reg = Logistic Regression, MLP = Multilayer Perceptron, RF = Random Forest"

png(paste("images\\f1_scores\\", "binary_f1_scores.png", sep=""), width=839, height=537)

results <- ggplot(df, aes(x = `ML.Classifier`, y = F1_Score, fill = `Feature.Type`)) +
  geom_col(position = "dodge",  alpha = 0.5) +
  geom_text(
    aes(label=scales::number(F1_Score, accuracy=0.01)),
    colour = "black", size = 6,
    vjust = 1.5, position = position_dodge(.9)
  ) +
  labs(fill = "Feature Type", caption=caption) +
  ggtitle("Binary Classifier F1-Scores on UNSW-NB15 Dataset") +
  xlab("Classifier Type") + 
  ylab("Accuracy (%)") +
  scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
  theme_classic() +
  scale_fill_manual(values = color_blind_palette) +
  theme(axis.title.x = element_text(size=13, face="bold", margin=margin(t = 10)), 
        axis.title.y = element_text(size=13, face="bold", margin=margin(r = 10)),
        axis.text.x = element_text(size=15),
        axis.text.y = element_text(size=12),
        plot.title = element_text(size=15, hjust=0.5))

# give output

print(results)

dev.off()

