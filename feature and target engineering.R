# Helper packages
library(tidyverse)
library(ggplot2)
library(visdat)
library(kableExtra)

# Feature engineering packages
library(caret)    # for various ML tasks
library(recipes)  # for feature engineering

library(rsample)  # for data splitting

# create ames training data
set.seed(123)
ames <- AmesHousing::make_ames()
split <- initial_split(ames, prop = 0.7, strata = "Sale_Price")
ames_train <- training(split)
ames_test <- testing(split)

# TARGET ENGINEERING
transformed_response <- log(ames_train$Sale_Price)

# Log transformation
ames_recipe <- recipe(Sale_Price ~.,
                      data = ames_train) %>%
  step_log(all_outcomes())
ames_recipe
log(-0.5)
log1p(-0.5)

# Log transformation
train_log_y <- log(ames_train$Sale_Price)
test_log_y <- log(ames_train$Sale_Price)

# Box Cox transformation
lambda <- forecast::BoxCox.lambda(ames_train$Sale_Price)
train_bc_y <- forecast::BoxCox(ames_train$Sale_Price, lambda = lambda)
test_bc_y <- forecast::BoxCox(ames_train$Sale_Price, lambda)

# Plot the differences
levels <- c("Normal", "Log_Transform", "BoxCox_Transform")
data.frame(
  Normal = ames_train$Sale_Price,
  Log_Transform <- train_log_y,
  BoxCox_Transform <- train_bc_y
) %>%
  gather(Transform, Value) %>%
  mutate(Transform = factor(Transform, levels = levels)) %>%
  ggplot(aes(Value, fill = Transform)) +
     geom_histogram(show.legend = FALSE, bins = 40) +
     facet_wrap(~ Transform, scales = "free_x")

# Box Cox transform a value
y <- forecast::BoxCox(10, lambda)

# Inverse Box Cox function
inv_box_cox <- function(x, lambda) {
  # for Box-Cox, lambda = 0 --> log transform
  if (lambda == 0) exp(x) else (lambda * x + 1)^(1 / lambda)
}

# Undo Box Cox transformation
inv_box_cox(y, lambda)

# DEALING WITH MISSINGNESS

# Set the graphical theme
ggplot2::theme_set(ggplot2::theme_light())

sum(is.na(AmesHousing::ames_raw))

# Visualization of missing values
AmesHousing::ames_raw %>%
  is.na() %>%
  reshape2::melt() %>%
  ggplot(aes(Var2, Var1, fill=value)) +
    geom_raster() +
    coord_flip() +
    scale_y_continuous(NULL, expand = c(0, 0)) +
    scale_fill_grey(name = "",
                    labels = c("Present", "Missing")) +
    xlab("Observation")

AmesHousing::ames_raw %>%
  filter(is.na('Garage Type')) %>%
  select('Garage Type', 'Garage Cars', 'Garage Area')

# Using visdat package to visualize missing values
vis_miss(AmesHousing::ames_raw, cluster = TRUE)

# IMPUTATION
# Impute missing values onto ames_recipe for Gr_Liv_Area
ames_recipe %>%
  step_medianimpute(Gr_Liv_Area)

# K-nearest neighbor imputation
ames_recipe %>%
  step_knnimpute(all_predictors(), neighbors = 6)

# Tree-based imputation
ames_recipe %>%
  step_impute_bag(all_predictors())  # or use step_bagimpute()

# Plot the different imputations
# 1) Actuals
impute_ames <- ames_train
set.seed(123)
index <- sample(seq_len(impute_ames$Gr_Liv_Area), 50)
index
actuals <- ames_train[index, ]
impute_ames$Gr_Liv_Area[index] <- NA
# Visualize the actual imputation
p1 <- ggplot() +
  geom_point(data = impute_ames, aes(Gr_Liv_Area, Sale_Price), alpha = .2) +
  geom_point(data = actuals, aes(Gr_Liv_Area, Sale_Price), color = "red") +
  scale_x_log10(limits = c(300, 5000)) +
  scale_y_log10(limits = c(10000, 500000)) +
  ggtitle("Actual Values")

# 2) Mean Imputation
mean_juiced <- recipe(Sale_Price ~., data = impute_ames) %>%
  step_meanimpute(Gr_Liv_Area) %>%
  prep(training = impute_ames, retain = TRUE) %>%
  juice()
mean_impute <- mean_juiced[index, ]
# Visualize mean imputation
p2 <- ggplot() +
  geom_point(data = actuals, aes(Gr_Liv_Area, Sale_Price), color = "red") +
  geom_point(data = mean_impute, aes(Gr_Liv_Area, Sale_Price), color = "blue") +
  scale_x_log10(limits = c(300, 5000)) +
  scale_y_log10(limits = c(10000, 500000)) +
  ggtitle("Mean Imputation")

# 3) KNN Imputation
knn_juiced <- recipe(Sale_Price ~., data = impute_ames) %>%
  step_knnimpute(Gr_Liv_Area) %>%
  prep(training = impute_ames, retain = TRUE) %>%
  juice()
knn_impute <- knn_juiced[index, ]
# Visualize knn imputation
p3 <- ggplot() +
  geom_point(data = actuals, aes(Gr_Liv_Area, Sale_Price), color = "red") +
  geom_point(data = knn_impute, aes(Gr_Liv_Area, Sale_Price), color = "blue") +
  scale_x_log10(limits = c(300, 5000)) +
  scale_y_log10(limits = c(10000, 500000)) +
  ggtitle("KNN Imputation")

# 4) Tree Based (Bagged) Imputation
bag_juiced <- recipe(Sale_Price ~., data = impute_ames) %>%
  step_bagimpute(Gr_Liv_Area) %>%
  prep(training = impute_ames, retain = TRUE) %>%
  juice()
bag_impute <- bag_juiced[index, ]
# Visualize the bagged imputation
p4 <- ggplot() +
  geom_point(data = actuals, aes(Gr_Liv_Area, Sale_Price), color = "red") +
  geom_point(data = bag_impute, aes(Gr_Liv_Area, Sale_Price), color = "blue") +
  scale_x_log10(limits = c(300, 5000)) +
  scale_y_log10(limits = c(10000, 500000)) +
  ggtitle("Bagged Imputation")

gridExtra::grid.arrange(p1, p2, p3, p4, nrow = 2)

# FEATURE FILTERING
caret::nearZeroVar(ames_train, saveMetrics = TRUE) %>%
  rownames_to_column() %>%
  filter(nzv)
# We can add step_zv() and step_nzv() to our ames_recipe to remove zero or
# near zero variance features.
ames_recipe %>%
  step_nzv(all_predictors())

# NUMERIC FEATURE ENGINEERING
# 1) Skewness
# Normalize all numeric columns
recipe(Sale_Price ~., data = ames_train) %>%
  step_YeoJohnson(all_numeric())

recipe(Sale_Price ~., data = ames_train) %>%
  step_BoxCox(all_numeric())

# 2) Standardization
# Standardizing features includes centering and scaling so
# that numeric variables have zero mean and unit variance, which provides a
# common comparable unit of measure across all the variables.
ames_recipe %>%
  step_center(all_numeric(), -all_outcomes()) %>%
  step_scale(all_numeric(), -all_outcomes())

# CATEGORICAL FEATURE ENGINEERING
# 1) Lumping
names(ames_train)
count(ames_train, Neighborhood) %>% arrange(n)
count(ames_train, Screen_Porch) %>% arrange(n)

# Lump levels for two features
lumping <- recipe(Sale_Price ~., data = ames_train) %>%
  step_other(Neighborhood, threshold = 0.01,
             other = "other") %>%
  step_other(Screen_Porch, threshold = 0.1,
             other = ">0")

# Apply this blue print --> you will learn about this later
apply_2_training <- prep(lumping, training = ames_train) %>%
  bake(ames_train)

# New distribution of Neighborhood
count(apply_2_training, Neighborhood) %>% arrange(n)
# New distribution of Screen_Porch
count(apply_2_training, Screen_Porch) %>% arrange(n)

# 2) One-hot & dummy encoding
# By default, step_dummy() will create a full rank encoding but you can change this
# by setting one_hot = TRUE.
# Lump levels for two features
recipe(Sale_Price ~., data = ames_train) %>%
  step_dummy(all_nominal(), one_hot = TRUE)

# If you have a data set with many categorical variables and those categorical variables 
# in turn have unique levels, the number of features can explode.
# In these cases you may want to explore label/ordinal encoding or some other alternatives
# 3) Label Encoding
# Original categories
count(ames_train, MS_SubClass)

# Label encoded
recipe(Sale_Price ~., data = ames_train) %>%
  step_integer(MS_SubClass) %>%
  prep(ames_train) %>%
  bake(ames_train) %>%
  count(MS_SubClass)

ames_train %>%
  select(contains("Qual"))
# The various xxx_Qual features in the Ames housing are not ordered factors.
# For ordered factors you could also use step_ordinalscore().
# Original Categories
count(ames_train, Overall_Qual)

# Label encoded
recipe(Sale_Price ~., data = ames_train) %>%
  step_integer(Overall_Qual) %>%
  prep(ames_train) %>%
  bake(ames_train) %>%
  count(Overall_Qual)

# DIMENSION REDUCTION
recipe(Sale_Price ~., data = ames_train) %>%
  step_center(all_numeric()) %>%
  step_scale(all_numeric()) %>%
  step_pca(all_numeric(), threshold = .95)

# PROPER IMPLEMENTATION
# Putting all the process together
blueprint <- recipe(Sale_Price ~., data = ames_train) %>%
  step_nzv(all_nominal()) %>%
  step_integer(matches("Qual|Cond|QC|Qu")) %>%
  step_center(all_numeric(), -all_outcomes()) %>%
  step_scale(all_numeric(), -all_outcomes()) %>%
  step_pca(all_numeric(), -all_outcomes())
blueprint
# Next train the blueprint on some training data
prepare <- prep(blueprint, training = ames_train)
prepare
# Lastly, we can apply our blueprint to new data (e.g., the training data or
# future test data) with bake().
baked_train <- bake(prepare, new_data = ames_train)
baked_test <- bake(prepare, new_data = ames_test)

# Consequently, the goal is to develop our blueprint, then within each resample
# iteration we want to apply prep() and bake() to our resample training and
# validation data.
blueprint <- recipe(Sale_Price ~., data = ames_train) %>%
  step_nzv(all_nominal()) %>%
  step_integer(matches("Qual|Cond|QC|Qu")) %>%
  step_center(all_numeric(), -all_outcomes()) %>%
  step_scale(all_numeric(), -all_outcomes()) %>%
  step_dummy(all_nominal(), -all_outcomes(), one_hot = TRUE)

# Specify resampling plan
cv <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 5
)

# Construct grid of hyperparameter values
hyper_grid <- expand.grid(k = seq(2, 25, by = 1))

# Tune a knn model using grid search
knn_fit2 <- train(
  blueprint,
  data = ames_train,
  method = "knn",
  trControl = cv,
  tuneGrid = hyper_grid,
  metric = "RMSE"
)

# print model results
knn_fit2

# plot cross validation results
ggplot(knn_fit2)
