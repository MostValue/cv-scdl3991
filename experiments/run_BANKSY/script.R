library(BASS)
library(Matrix)
library(Seurat)
library(ggplot2)
library(tidyverse)



data_path <- file.path('C:/Users/Kerui Yang/github/cv-scdl3991/data/')
output_path <- file.path('C:/Users/Kerui Yang/github/cv-scdl3991/outputs/BANKSY/')



if(!dir.exists(file.path(output_path))){
  dir.create(file.path(output_path), recursive = TRUE)
}


data_files <-  list.files(data_path, full.names = TRUE, recursive = TRUE)
print(data_files)

# Selecting the data files:

# MOUSE DATA

filename = "C:/Users/Kerui Yang/github/cv-scdl3991/data/MH/Moffitt_and_Bambah-Mukku_et_al_merfish_all_cells.csv"
result = read.csv(filename, header = TRUE, row.names = 1) 

result <- result %>% select(-c(1:9))

colnames(result)
str(result)
rownames(result)

# Creating seurat obj


obj = CreateSeuratObject(counts = t(result))


Assays(obj)
    




