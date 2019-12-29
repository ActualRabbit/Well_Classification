###--------------------------------------------------------------------------------------------------------------------------------------------
# Classifying Well Status - Untapped Oil 2019 
###--------------------------------------------------------------------------------------------------------------------------------------------

set.seed(3801)

## Load Packages ------------------------------------------------------------------------------------------------------------------------------

library(mlr)
library(tidyverse)
library(lubridate)
library(fastDummies)
library(xgboost)

## Read in Raw Data ---------------------------------------------------------------------------------------------------------------------------

Raw_Train <- read_csv("~/Header_train.csv", col_names = TRUE)

Raw_Validation <- read_csv("~/Header_validation.csv", col_names = TRUE)

Raw_Test <- read_csv("~/Header_test.csv", col_names = TRUE)

Well_Class_Train_Raw <- read_csv("~/Well_class_train.csv", col_names = TRUE)

Well_Class_Validation_Raw <- read_csv("~/Well_class_validate.csv", col_names = TRUE)


## Joining in Well Classes and Binding Training and Validation (for Custom Splits) --------------------------------------------------------------

Joined_Train <- left_join(Raw_Train, Well_Class_Train_Raw, by = "EPAssetsId")

Joined_Validation <- left_join(Raw_Validation, Well_Class_Validation_Raw, by = "EPAssetsId")


## Changing Dates into Days, Imputing NAs, Removing Some Columns, and Creating a Flag if the Licensee and Current Operator are Different Firms ---

is_na_value <- function(x) x %in% c("Not Available", "<NA>")

days_from_origin <- function(y) {as.numeric(interval(ymd("1883-01-01"), mdy_hms(y)), "days")}

features_to_be_cleaned <- c("EPAssetsId", "WellName", "WellNameAmended", "SurveySystem", "Surf_Location", "Surf_Township", "Surf_Meridian", 
                            "Surf_Range", "Surf_Section", "Surf_LSD", "Surf_TownshipRange", "Surf_QuarterUnit", "Surf_Unit", 
                            "Surf_Block", "Surf_NTSMapSheet", "Surf_Series", "Surf_Area", "Surf_Sheet", "Surf_QuarterSection",  
                            "BH_Location", "BH_TownshipRange", "BH_QuarterUnit", "BH_Unit", "BH_Block", "BH_NTSMapSheet",  
                            "BH_Series", "BH_Area", "BH_Sheet", "BH_QuarterSection", "BH_Township", "BH_Meridian", "BH_Range",  
                            "BH_Section", "BH_LSD", "Country", "RegulatoryAgency", "PSACAreaName", "Municipality",  
                            "LicenseeID", "CurrentOperatorID", "WellTypeStandardised", "UnitName", "UnitFlag", "CompletionEvents",
                            "UWI", "LicenceNumber", "StatusSource")

interesting_features_dropped <- c("Agent", "CurrentOperatorParent", "LicenseeParentCompany", "SurfaceOwner",
                                  "Pool", "Licensee", "DrillingContractor", "CurrentOperator", "Field")

date_columns <- c("LicenceDate", "ConfidentialReleaseDate", "SpudDate", "FinalDrillDate", 
                  "RigReleaseDate", "StatusDate", "CompletionDate", "SurfAbandonDate")

Training_Data <- bind_rows(Joined_Train, Joined_Validation, .id = "PARENT_TABLE_NAME") %>%    # Union of Train and Validation
  mutate_all(funs(ifelse(is_na_value(.), NA, .))) %>% 
  mutate_at(date_columns, days_from_origin) %>% 
  mutate(Operator_Transfer = case_when(Licensee == CurrentOperator ~ 1, Licensee != CurrentOperator ~ 0)) %>% 
  select(-PARENT_TABLE_NAME, -features_to_be_cleaned, -interesting_features_dropped)


Test_Data <- Raw_Test %>% mutate_all(funs(ifelse(is_na_value(.), NA, .))) %>% 
  mutate_at(date_columns, days_from_origin) %>%
  mutate(Operator_Transfer = case_when(Licensee == CurrentOperator ~ 1, Licensee != CurrentOperator ~ 0)) %>% 
  select(-features_to_be_cleaned, -interesting_features_dropped)

rm(features_to_be_cleaned, interesting_features_dropped, date_columns)


## Remove Extraneous Dataframes from the Global Environment -----------------------------------------------------------------------------------

rm(Raw_Train, Raw_Validation, Well_Class_Train_Raw, Well_Class_Validation_Raw,
   Joined_Train, Joined_Validation)


## Create Dummy Columns for the Categorical Variables ------------------------------------------------------------------------------------------

factors_to_dummies <- c("WellType", "Formation", "LaheeClass", "Confidential", 
                        "OSArea", "OSDeposit", "WellProfile","WellSymbPt1",
                        "PSACAreaCode", "UnitID", "OpenHole", "Province")

Training_Data <- dummy_columns(Training_Data, select_columns = factors_to_dummies) %>% select(-factors_to_dummies)

Test_Data <- dummy_columns(Test_Data, select_columns = factors_to_dummies) %>% select(-factors_to_dummies)

rm(factors_to_dummies)


## Select Only the Intersecting Vectors -------------------------------------------------------------------------------------------------------

Training_colnames <- colnames(Training_Data)

Test_colnames <- colnames(Test_Data)

intersecting_vectors <- dplyr::intersect(Training_colnames, Test_colnames)

Training_Data <- Training_Data %>% select(intersecting_vectors, well_status_code)

Test_Data <- Test_Data %>% select(intersecting_vectors)

rm(Training_colnames, Test_colnames, intersecting_vectors)


## Normalize the Numeric Vectors --------------------------------------------------------------------------------------------------------------

columns_for_normalization <- c("DaysDrilling", "DrillMetresPerDay", "TVD", "NumberofWells", "MaxProd_BOE", "FractureStages")

Training_Data <- normalizeFeatures(Training_Data, method = "standardize", cols = columns_for_normalization)

Test_Data <- normalizeFeatures(Test_Data, method = "standardize", cols = columns_for_normalization)

rm(columns_for_normalization)


## Re-Sample the Training and Validation Data -------------------------------------------------------------------------------------------------

Training_Data <- Training_Data %>% mutate(ID = row_number()) 

Training_Resampled <- Training_Data %>% sample_frac(0.7) 

Validation_Resampled <- anti_join(Training_Data, Training_Resampled, by = 'ID')

Training_Matrix <- Training_Resampled %>% select(-ID, -well_status_code)

Training_Label <- Training_Resampled %>% select(well_status_code)

Validation_Matrix <- Validation_Resampled %>% select(-ID, -well_status_code)

Validation_Label <- Validation_Resampled %>% select(well_status_code)

Training_Data <- Training_Data %>% select(-ID)

Full_Training_Matrix <- Training_Data %>% select(-well_status_code)

Full_Training_Label <- Training_Data %>% select(well_status_code)

rm(Training_Resampled, Validation_Resampled)


## Convert to XGB Matrix ----------------------------------------------------------------------------------------------------------------------

tibble_to_xgb_DMatrix <- function(matrix, label) {
  Matrix <- as.matrix(matrix)
  Label <- as.matrix(label)
  xgb.DMatrix(data = Matrix, label = Label)
}

Training_XGB <- tibble_to_xgb_DMatrix(matrix = Training_Matrix, label = Training_Label)

Validation_XGB <- tibble_to_xgb_DMatrix(matrix = Validation_Matrix, label = Validation_Label)

Full_Training_XGB <- tibble_to_xgb_DMatrix(matrix = Full_Training_Matrix, label = Full_Training_Label)

rm(Training_Matrix, Training_Label, Validation_Matrix, Full_Training_Matrix, Full_Training_Label)


## XGBoost Model using Training and Validation Data -------------------------------------------------------------------------------------------

parameters <- list(
  "booster" = "gbtree",
  "num_class" = 3,
  "objective" = "multi:softprob",
  "eval_metric" = "mlogloss",
  "eta" = 0.7
)

watchlist <- list(train = Training_XGB, evaluate = Validation_XGB)

Model_XGB <- xgb.train(
  params = parameters,
  data = Training_XGB,
  nrounds = 10,
  watchlist
)


## Generate Validation Predictions and Calculate Validation Accuracy --------------------------------------------------------------------------

Validation_Predictions <- as.data.frame(predict(Model_XGB, Validation_XGB, reshape = TRUE))

Validation_Accuracy <- bind_cols(as.data.frame(Validation_Label), Validation_Predictions) %>% 
  mutate(predicted_well_status_code = case_when(V1 > V2 & V1 > V3 ~ 0,
                                      V2 > V1 & V2 > V3 ~ 1,
                                      V3 > V1 & V3 > V2 ~ 2)) %>% 
  mutate(Correct_Prediction_Flag = case_when(well_status_code == predicted_well_status_code ~ 1, 
                                             well_status_code != predicted_well_status_code ~ 0))

cat("Validation Acccuracy:", 
    round((sum(Validation_Accuracy$Correct_Prediction_Flag)/length(Validation_Accuracy$Correct_Prediction_Flag))*100, 2), "%")


## Re-train XGB Model with the Full Training Data ---------------------------------------------------------------------------------------------

Model_XGB_Test_Ready <- xgb.train(
  params = parameters,
  data = Full_Training_XGB,
  nrounds = 10,
  watchlist
)


## Build Test Predictions and Write Out for Submission ----------------------------------------------------------------------------------------

Test_XGB <- xgb.DMatrix(as.matrix(Test_Data))

Test_Predictions_10R <- as.data.frame(predict(Model_XGB_Test_Ready, Test_XGB, reshape = TRUE))

Test_EPA_ID <- Raw_Test %>% select(EPAssetsId)

Output_File <- bind_cols(Test_EPA_ID, Test_Predictions_10R) %>% 
  mutate(well_status_code = case_when(V1 > V2 & V1 > V3 ~ 0,
                                      V2 > V1 & V2 > V3 ~ 1,
                                      V3 > V1 & V3 > V2 ~ 2)) %>% select(-V1, -V2, -V3)

write_csv(Output_File, "~/Output_File.csv")
