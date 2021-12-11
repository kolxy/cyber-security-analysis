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

df <- read.csv("results.csv",
               header=TRUE,
               sep=",",
               stringsAsFactors = TRUE)

# convert accuracy to percentages

df <- df %>% mutate(Accuracy = Accuracy * 100, 
                    ML.Classifier = ifelse(ML.Classifier == "K-Nearest Neighbors", "KNN",
                                           ifelse(ML.Classifier == "Logistic Regression", "Log. Reg",
                                                  ifelse(ML.Classifier == "Multilayer Perceptron", "MLP", "RF"))))

# take care of binary data

bin_res <- df %>% 
  filter(Class.Type == "Binary") %>%
  select(Feature.Type, Accuracy, ML.Classifier) %>%
  arrange(Accuracy)


# take care of multiclass data

mul_res <- df %>% 
  filter(Class.Type == "Multiclass") %>%
  select(Feature.Type, Accuracy, ML.Classifier) %>%
  arrange(Accuracy)

# plot binary results

caption <- "KNN = K-Nearest Neighbors, Log. Reg = Logistic Regression, MLP = Multilayer Perceptron, RF = Random Forest"

png(paste("images\\accuracy_scores\\", "binary_accuracy.png", sep=""), width=839, height=537)

binary_results <- ggplot(bin_res, aes(x = `ML.Classifier`, y = Accuracy, fill = `Feature.Type`)) +
  geom_col(position = "dodge",  alpha = 0.5) +
  geom_text(
    aes(label=scales::number(Accuracy, accuracy=0.01)),
    colour = "black", size = 6,
    vjust = 1.5, position = position_dodge(.9)
  ) +
  labs(fill = "Feature Type", caption=caption) +
  ggtitle("Classifier Accuracy on UNSW-NB15 Dataset - Binary Labels") +
  xlab("Classifier Type") + 
  ylab("Accuracy (%)") +
  scale_y_continuous(limits = c(0, 100), expand = c(0, 0)) +
  theme_classic() +
  scale_fill_manual(values = color_blind_palette) +
  theme(axis.title.x = element_text(size=13, face="bold", margin=margin(t = 10)), 
        axis.title.y = element_text(size=13, face="bold", margin=margin(r = 10)),
        axis.text.x = element_text(size=15),
        axis.text.y = element_text(size=12),
        plot.title = element_text(size=15, hjust=0.5))

# give output

print(binary_results)

dev.off()

# plot multiclass results

png(paste("images\\accuracy_scores\\", "multiclass_accuracy.png", sep=""), width=839, height=537)

multiclass_results <- ggplot(mul_res, aes(x = ML.Classifier, y = Accuracy, fill = Feature.Type)) +
  geom_col(position = "dodge", alpha = 0.5) +
  geom_text(
    aes(label = scales::number(Accuracy, accuracy=0.01)),
    colour = "black", size = 6,
    vjust = 1.5, position = position_dodge(.9)
  ) +
  labs(fill = "Feature Type",  caption=caption) +
  ggtitle("Classifier Accuracy on UNSW-NB15 Dataset - Multiclass Labels") +
  xlab("Classifier Type") + 
  ylab("Accuracy (%)") +
  scale_y_continuous(limits = c(0, 100), expand = c(0, 0)) +
  theme_classic() + 
  scale_fill_manual(values = color_blind_palette) +
  theme(axis.title.x = element_text(size=13, face="bold", margin=margin(t = 10)), 
        axis.title.y = element_text(size=13, face="bold", margin=margin(r = 10)),
        axis.text.x = element_text(size=15),
        axis.text.y = element_text(size=12),
        plot.title = element_text(size=15, hjust=0.5))


# give output

print(multiclass_results)

dev.off()


