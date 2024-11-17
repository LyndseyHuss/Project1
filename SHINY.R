# MIDSEMESTER PROJECT 410
# New Sentiment Analysis with AFINN Lexicon

# --------------------------- Libraries and Seed --------------------------
# Load necessary libraries
library(readxl)        # For reading Excel files
library(dplyr)         # For data manipulation
library(rsample)       # For data splitting
library(randomForest)  # For Random Forest model
library(shiny)

# Set seed for reproducibility
set.seed(410)

# --------------------------- Data Import --------------------------
# Read the Excel file into R


library(readxl)

# Specify the full path to the file
lotwize <- read_excel("/Users/lyndseyhuss/Documents/MGSC_410/MGSC_410/Project1/lotwize_case.xlsx")

# --------------------------- Initial Exploration ------------------
# Get the dimensions of the dataset
dim(lotwize)

# Get a glimpse of the dataset structure
glimpse(lotwize)

# Check the number of NA values in 'resoFacts/stories'
sum(is.na(lotwize$`resoFacts/stories`))

# --------------------------- Preprocessing ------------------------

# Function to convert specified columns to factors
convert_to_factors <- function(df, cols) {
  df %>% mutate(across(all_of(cols), as.factor))
}

# Columns to convert to factors
factor_cols <- c("city", "homeType", "bathrooms", "bedrooms")
lotwize <- convert_to_factors(lotwize, factor_cols)

# Function to convert specified columns to numeric
numeric_conversion <- function(df, cols) {
  df %>%
    mutate(across(all_of(cols), ~ as.numeric(as.character(.))))
}

# Convert relevant columns to numeric
lotwize <- numeric_conversion(lotwize, c("price", "yearBuilt", "latitude", "longitude"))

# --------------------------- Feature Engineering ------------------

# Calculate the age of each home
lotwize <- lotwize %>%
  filter(!is.na(yearBuilt)) %>%
  mutate(age = 2024 - yearBuilt)

# Convert specific columns to logical
logical_cols <- c("resoFacts/hasView", "resoFacts/hasSpa", "resoFacts/canRaiseHorses")
lotwize <- lotwize %>%
  mutate(across(all_of(logical_cols), ~ case_when(
    . == 'TRUE' ~ TRUE,
    . == 'FALSE' ~ FALSE,
    TRUE ~ NA
  )))

# Create the 'luxury' variable based on specified conditions
lotwize <- lotwize %>%
  mutate(
    luxury = (
      rowSums(select(., all_of(logical_cols)), na.rm = TRUE) > 0 |
        `resoFacts/garageParkingCapacity` > 2 |
        `resoFacts/fireplaces` > 1 |
        `resoFacts/stories` %in% c(2, 3, 5)
    ),
    luxury = as.factor(luxury)
  )

# --------------------------- Exploratory Data Analysis -----------
# Function to remove outliers based on IQR
remove_outliers <- function(df, feature) {
  Q1 <- quantile(df[[feature]], 0.25, na.rm = TRUE)
  Q3 <- quantile(df[[feature]], 0.75, na.rm = TRUE)
  IQR_value <- IQR(df[[feature]], na.rm = TRUE)
  df %>% filter(df[[feature]] >= (Q1 - 1.5 * IQR_value) & df[[feature]] <= (Q3 + 1.5 * IQR_value))
}

# Remove outliers for important features
important_features <- c("price", "age")
lotwize <- reduce(important_features, remove_outliers, .init = lotwize)

# --------------------------- Create Unique Identifier ----------------
# Add a unique identifier to each row
lotwize <- lotwize %>%
  mutate(home_id = row_number())

# --------------------------- Sentiment Analysis Preparation --------------------
# Replace "NA" strings with actual NA values and handle them
lotwize$description <- na_if(lotwize$description, "NA")
lotwize$description[is.na(lotwize$description)] <- ""

# --------------------------- Train/Test Split ----------------------
# Perform stratified split based on 'city' to ensure representation
lotwize_split <- initial_split(lotwize, prop = 0.8, strata = city)
lotwize_train <- training(lotwize_split)
lotwize_test  <- testing(lotwize_split)

# --------------------------- Determine Top 50 Cities from Training Set ------------
# Determine the top 50 cities based on the training data
top50_cities <- lotwize_train %>%
  count(city, sort = TRUE) %>%
  slice_head(n = 50) %>%
  pull(city) %>%
  as.character()

# Inspect the top50_cities
print(top50_cities)

# --------------------------- Handle Categorical Variables --------
# Define the mapping function
handle_categorical <- function(df, top_categories) {
  df %>%
    mutate(
      city = as.character(city),  # Convert to character to prevent factor issues
      city = ifelse(city %in% top_categories, city, "Other"),  # Map to top categories or "Other"
      city = factor(city, levels = c(top_categories, "Other"))  # Convert back to factor with specified levels
    )
}

# Apply the mapping to the training set
lotwize_train_clean <- handle_categorical(lotwize_train, top50_cities)

# --------------------------- Drop Unused Factor Levels -------------------
# Apply droplevels to remove any unused levels in all factor variables
lotwize_train_clean <- lotwize_train_clean %>%
  mutate(across(where(is.factor), fct_drop))

lotwize_test_clean <- lotwize_test_clean %>%
  mutate(across(where(is.factor), fct_drop))

# --------------------------- Verify 'city' Distribution After Mapping
# Check the distribution of 'city' in the training set after mapping
lotwize_train_clean %>%
  count(city) %>%
  arrange(desc(n)) %>%
  print(n = 50)

# --------------------------- Sentiment Analysis on Training Data --------------------
# --------------------------- **Replaced Bing with AFINN Sentiment Lexicon** --------------------
# Load AFINN sentiment lexicon
afinn_sentiments <- get_sentiments("afinn")

# Tokenize and join with AFINN lexicon for training data
sentiment_scores_train <- lotwize_train_clean %>%
  select(home_id, description) %>%
  unnest_tokens(word, description) %>%
  inner_join(afinn_sentiments, by = "word") %>%
  group_by(home_id) %>%
  summarise(sentiment_score = sum(value, na.rm = TRUE))  # Sum AFINN sentiment scores

# Merge sentiment scores with the training dataset
lotwize_train_clean <- lotwize_train_clean %>%
  left_join(sentiment_scores_train, by = "home_id") %>%
  mutate(
    sentiment_score = if_else(is.na(sentiment_score), 0, sentiment_score)  # Replace NA with 0 sentiment
  )
# Define top 50 cities for input options
top50_cities <- c(
  "Bakersfield", "Fresno", "Long Beach", "San Francisco", "San Diego",
  "Los Angeles", "Anaheim", "Other"  # Keep most relevant cities only
)

# ------------------- UI -------------------
ui <- fluidPage(
  titlePanel("Automated Valuation Model (AVM)"),
  
  sidebarLayout(
    sidebarPanel(
      selectInput("city", "City", choices = top50_cities),
      numericInput("bathrooms", "Number of Bathrooms", value = 2, min = 0),
      numericInput("bedrooms", "Number of Bedrooms", value = 3, min = 0),
      numericInput("age", "Age of the Property", value = 10, min = 0),
      numericInput("latitude", "Latitude", value = 33.6846),
      numericInput("longitude", "Longitude", value = -117.8265),
      selectInput("homeType", "Home Type", choices = c(
        "Apartment" = "APARTMENT",
        "Condo" = "CONDO",
        "Single Family" = "SINGLE_FAMILY",
        "Townhouse" = "TOWNHOUSE",
        "Other" = "OTHER"
      )),
      selectInput("luxury", "Is it a luxury property?", choices = c("Yes" = TRUE, "No" = FALSE)),
      numericInput("sentiment_score", "Sentiment Score", value = 0),
      actionButton("predict", "Generate Prediction")
    ),
    mainPanel(
      h3("Predicted Property Price"),
      textOutput("prediction"),
      h4("Recommended Similar Properties"),
      tableOutput("recommendations")
    )
  )
)

# ------------------- Server -------------------
server <- function(input, output, session) {
  observeEvent(input$predict, {
    req(input$city, input$bathrooms, input$bedrooms, input$age, input$latitude, input$longitude)
    
    # Prepare user input data
    user_data <- data.frame(
      city = factor(input$city, levels = levels(lotwize$city)),
      bathrooms = as.numeric(input$bathrooms),
      bedrooms = as.numeric(input$bedrooms),
      age = as.numeric(input$age),
      latitude = as.numeric(input$latitude),
      longitude = as.numeric(input$longitude),
      homeType = factor(input$homeType, levels = levels(lotwize$homeType)),
      luxury = factor(input$luxury, levels = c(TRUE, FALSE)),
      sentiment_score = as.numeric(input$sentiment_score)
    )
    
    tryCatch({
      # Generate prediction
      prediction <- predict(avm_model, newdata = user_data)
      predicted_price <- exp(prediction)  # Revert log transformation
      output$prediction <- renderText(paste("Predicted Price: $", round(predicted_price, 2)))
      
      # Generate recommendations
      recommendations <- lotwize %>%
        filter(
          city == user_data$city,
          abs(price - predicted_price) <= 50000  # Adjust price range as needed
        ) %>%
        arrange(abs(price - predicted_price)) %>%
        head(5) %>%
        select(city, price, bathrooms, bedrooms, homeType, age)  # Columns to display
      output$recommendations <- renderTable(recommendations)
      
    }, error = function(e) {
      output$prediction <- renderText("An error occurred during prediction. Please check inputs.")
    })
  })
}

# ------------------- Run Shiny App -------------------
shinyApp(ui = ui, server = server)
