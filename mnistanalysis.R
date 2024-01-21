library(dslabs)
library(MASS)
library(ggplot2)

mnist <- read_mnist()

X <- mnist$train$images
y <- mnist$train$labels


#PCA 
#preprocessing step to remove white/blank space

avg_pixels <- apply(X, 2, mean) 

applied_preproc <- avg_pixels > 0.2 # 0.2 is our threshold, so any pixel value with less than that will be presumed to be whitespace or not as important

X_preprocessed <- X[, applied_preproc]


```


```{r}

s <- svd(X_preprocessed)

U <- s$u
D <- s$d # singular values, gives insight into how much variance is captured per component
V <- s$v

```

```{r}

# V2 <- V[,c()] # how to choose top principal components?

variance_explained <- D^2 / sum(D^2)
cumulative_variance <- cumsum(variance_explained)

# By calculating how much variance each component captures, and plotting the cumulative sum of variances, you can identify how many components to consider as principal components.

plot(cumulative_variance, xlab = "Number of Components", ylab = "Cumulative Variance Explained", type = 'b')

# elbow plot levels off at around 100 components (cumulative variance already above 90% at this point), so 

pc <- 100 # modify this value to change the number of principal components to consider

reduced_components <- V[, 1:pc]


C <- X_preprocessed %*% reduced_components # C holds the transformed data post PCA, so this can then be used for further analysis like clustering or input to a model

cvec <- y

colors <- rainbow(length(unique(cvec))) # Generate distinct colors

plot(C, col = colors[cvec + 1], asp = 1)
```

```{r}

legend("topright", # position of the legend
       legend = 0:9, # labels
       col = colors, # colors
       pch = 1, # type of point
       title = "Digit Labels")

```

```{r}

# testing C using pre-defined functions

X_test <- mnist$test$images
y_test <- mnist$test$labels

# X_preprocessed <- X[, applied_preproc]

# Apply the same preprocessing to the test data
X_preprocessed_test <- X_test[, applied_preproc]


dim(X)
dim(X_test)

dim(X_preprocessed_test)
dim(X_preprocessed)
dim(reduced_components)

# checking the shapes of stuff because I got lots of shape mismatches

```

```{r}

# testing how I did with predefined models

library(class)

C_test <- X_preprocessed_test %*% reduced_components

# Perform kNN classification
knn_model <- knn(train = C, test = C_test, cl = y, k = 3)

# Calculate and print accuracy
accuracy <- sum(knn_model == y_test) / length(y_test)
print(paste("Accuracy:", accuracy))

```


```{r}
# TODO: show an original sample, a PCA modified sample, and how it performs in a model (lin reg, lda, clustering?). Prof. also wanted to maybe see a subset of the data (like three or four digits, I am thinking of using 0, 1, 3, 8)

# 0:
label_of_0 <- y == 0
list_0 <- which(label_of_0)

# Selecting ten images
selected_indices <- list_0[1:5]

reconstructed_0 <- C[selected_indices, ] %*% t(reduced_components)

# Set up the plot area (5 images in one row)
par(mfrow = c(2, 5))

# Loop over the selected indices and display each image
for (index in selected_indices) {
  image_matrix <- matrix(X[index, ], nrow = 28, ncol = 28, byrow = TRUE)
  image(1:28, 1:28, image_matrix, col = grey.colors(256), main = paste("Image", index))
}

for (index in selected_indices) {
  reconstructed_image_matrix <- matrix(reconstructed_0, nrow = 28, ncol = 28, byrow = TRUE)
  image(1:28, 1:28, reconstructed_image_matrix, col = grey.colors(256), main = paste("Reconstructed Image", index))}


```

```{r}

# 1:
label_of_1 <- y == 1
list_1 <- which(label_of_1)

# Selecting ten images
selected_indices <- list_1[1:5]

reconstructed_1 <- C[selected_indices, ] %*% t(reduced_components)

# Set up the plot area (5 images in one row)
par(mfrow = c(2, 5))

# Loop over the selected indices and display each image
for (index in selected_indices) {
  image_matrix <- matrix(X[index, ], nrow = 28, ncol = 28, byrow = TRUE)
  image(1:28, 1:28, image_matrix, col = grey.colors(256), main = paste("Image", index))
}

for (index in selected_indices) {
  reconstructed_image_matrix <- matrix(reconstructed_1, nrow = 28, ncol = 28, byrow = TRUE)
  image(1:28, 1:28, reconstructed_image_matrix, col = grey.colors(256), main = paste("Reconstructed Image", index))}

```

```{r}

# 3:
label_of_3 <- y == 3
list_3 <- which(label_of_3)

# Selecting ten images
selected_indices <- list_3[1:5]

reconstructed_3 <- C[selected_indices, ] %*% t(reduced_components)

# Set up the plot area (5 images in one row)
par(mfrow = c(2, 5))

# Loop over the selected indices and display each image
for (index in selected_indices) {
  image_matrix <- matrix(X[index, ], nrow = 28, ncol = 28, byrow = TRUE)
  image(1:28, 1:28, image_matrix, col = grey.colors(256), main = paste("Image", index))
}

for (index in selected_indices) {
  reconstructed_image_matrix <- matrix(reconstructed_3, nrow = 28, ncol = 28, byrow = TRUE)
  image(1:28, 1:28, reconstructed_image_matrix, col = grey.colors(256), main = paste("Reconstructed Image", index))}

```

```{r}

# 8:
label_of_8 <- y == 8
list_8 <- which(label_of_8)

# Selecting ten images
selected_indices <- list_8[1:5]

reconstructed_8 <- C[selected_indices, ] %*% t(reduced_components)

# Set up the plot area (5 images in one row)
par(mfrow = c(2, 5))

# Loop over the selected indices and display each image
for (index in selected_indices) {
  image_matrix <- matrix(X[index, ], nrow = 28, ncol = 28, byrow = TRUE)
  image(1:28, 1:28, image_matrix, col = grey.colors(256), main = paste("Image", index))
}

for (index in selected_indices) {
  reconstructed_image_matrix <- matrix(reconstructed_8, nrow = 28, ncol = 28, byrow = TRUE)
  image(1:28, 1:28, reconstructed_image_matrix, col = grey.colors(256), main = paste("Reconstructed Image", index))}

```





#LDA

# Identify and remove constant variables
non_constant_cols <- apply(X, 2, function(x) length(unique(x)) > 1)
X <- X[, non_constant_cols]

# Combine features and labels into a data frame
data <- data.frame(X, Label = as.factor(y))

# Fit an LDA model
lda_model <- lda(Label ~ ., data = data)

means <- lda_model$means
prior_probs <- lda_model$prior
coefficients <- lda_model$scaling

lda_predictions <- predict(lda_model)


lda_data <- data.frame(
  LD1 = lda_predictions$x[, 1],  
  LD2 = lda_predictions$x[, 2],  
  Label = y  
)

ggplot(lda_data, aes(x = LD1, y = LD2, color = Label)) +
  geom_point() +
  labs(title = "LDA Scatterplot", x = "LD1", y = "LD2") +
  scale_color_brewer(palette = "Set1")


#Linear Regression 
images_train_lm <- images_train[1:10000]
labels_train_lm <-labels_train[1:10000]
images_test_lm <-images_test[1:10000]
labels_test_lm <-labels_test[1:10000]

pixelsds <-colSds(images_train_lm)
keep <-which(pixelsds >20)

p_hat_lm <-sapply(0:9, function(digit){
  y <-as.numeric(labels_train_lm == digit)
  fit_lm <-lm(y~images_train_lm[,keep])
  cbind(1, images_test_lm[,keep]) %*% fit_lm$coefficients
})

y_hat_lm <-apply(p_hat_lm, 1, which.max) - 1

mean(y_hat_lm ==labels_test_lm)


#Lasso 
images_train_lasso <- images_train[1:10000]
labels_train_lasso <-labels_train[1:10000]
images_test_lasso <-images_test[1:10000]
labels_test_lasso <-labels_test[1:10000]

images_train_lasso <-cbind(1, images_train_lasso)
images_test_lasso <-cbind(1, images_test_lasso)

p_hat_lasso <-sapply(0:9, function(digit){
  y <- as.numeric(labels_train_lasso ==digit)
  fit_lasso <- cv.glmnet(images_train_lasso, y, alpha=1, nfolds=10)
  best_lambda <- fit_lasso$lambda.min
  final_lasso <- glmnet(images_train_lasso, y, alpha=1, lambda=best_lambda)
  return(predict(final_lasso, s=best_lambda, newx=images_test_lasso))
})

y_hat_lasso <- apply(p_hat_lasso, 1, which.max) - 1 
mean(y_hat_lasso == labels_test_lasso)


#KNN Classification 
images_train_knn <- images_train[1:10000]
labels_train_knn <-labels_train[1:10000]
images_test_knn <-images_test[1:10000]
labels_test_knn <-labels_test[1:10000]

colnames(images_train_knn) <-1:ncol(images_train_knn)
colnames(images_test_knn) <-1:ncol(images_test_knn)

fit_knn <-train(y~ ., method="knn",
                data=data.frame(y=as.factor(labels_train_knn), images_train_knn),
                tuneGrid = data.frame(k=seq(5,13,2)),
                trControl = trainControl(method="cv", number=10, p=0.9))
bestK <- fit_knn$bestTune

final_knn <-knn3(y~.,
                 data=data.frame(y=as.factor(labels_train_knn), images_train_knn),
                 k=bestK)
y_hat_knn <- predict(final_knn,
                     newdata=data.frame(images_test_knn),
                     type="class")
        )


mean(y_hat_knn ==labels_test_knn)