RequiredPackages <- c("cvms", "ggplot2", "ggfittext")


for (pkg in RequiredPackages) {
  if (pkg %in% rownames(installed.packages()) == FALSE)
  {install.packages(pkg)}
  if (pkg %in% rownames(.packages()) == FALSE)
  {library(pkg, character.only = TRUE)}
}

setwd("C:/Users/waltz/Documents/cyber-security-analysis/output")

titles <- c("Logistic Regression",
            "Logistic Regression PCA",
            "Random Forest",
            "Random Forest PCA",
            "Multilayer Perceptron",
            "Multilayer Perceptron PCA",
            "K-Nearest Neighbors",
            "K-Nearest Neighbors PCA")

file_names <- c("logistic_regression",
                "logistic_regression_pca",
                "random_forest",
                "random_forest_pca",
                "mlp",
                "mlp_pca",
                "knn",
                "knn_pca")

# binary loop
for (val in 1:8) {
  image_name <- paste("images\\", "binary\\", file_names[val], "_binary_mat.png", sep="")
  title <- paste(titles[val], " Binary Confusion Matrix", sep="")
  df <- read.csv(paste("matrix_output\\", file_names[val], "_binary_mat.csv", sep=""))
  
  png(image_name)
  
  # funso's ggplot masterpiece :P
  plot <- ggplot(df, aes(x=target, y=prediction, fill=n, label=scales::comma(n))) + 
    geom_tile() + 
    theme_bw() +
    theme(axis.text = element_text(size=15), 
          axis.title=element_text(size=20), 
          plot.title = element_text(size=15, hjust=0.5),
          legend.position = "none") +
    coord_equal() + 
    xlab(label = "Target") +
    ylab(label = "Prediction") +
    scale_fill_distiller(palette="Blues", direction=1) + 
    scale_x_continuous(breaks = seq(0, 1, by=1)) + 
    scale_y_continuous(breaks = seq(0, 1, by=1)) + 
    labs(title = title) + # using a title instead 
    geom_fit_text(reflow=TRUE, grow=FALSE, contrast=TRUE, size=15) # printing values
  
  print(plot)
  dev.off()
}

# multiclass loop
for (val in 1:8) {
  image_name <- paste("images\\", "multiclass\\", file_names[val], "_multiclass_mat.png", sep="")
  title <- paste(titles[val], " Multiclass Confusion Matrix", sep="")
  df <- read.csv(paste("matrix_output\\", file_names[val], "_multiclass_mat.csv", sep=""))
  
  png(image_name)
  
  # funso's ggplot masterpiece :P
  plot <- ggplot(df, aes(x=target, y=prediction, fill=n, label=scales::comma(n))) + 
    geom_tile() + 
    theme_bw() +
    theme(axis.text = element_text(size=15), 
          axis.title=element_text(size=20), 
          plot.title = element_text(size=15, hjust=0.5),
          legend.position = "none") +
    coord_equal() + 
    xlab(label = "Target") +
    ylab(label = "Prediction") +
    scale_fill_distiller(palette="Blues", direction=1) + 
    scale_x_continuous(breaks = seq(0, 8, by=1)) + 
    scale_y_continuous(breaks = seq(0, 8, by=1)) + 
    labs(title = title) + # using a title instead 
    geom_fit_text(reflow=TRUE, grow=FALSE, contrast=TRUE, size=15) # printing values
  
  print(plot)
  dev.off()
}


