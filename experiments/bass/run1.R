library(BASS)
library(zellkonverter)
library(SingleCellExperiment)


# loading in data

adata = readH5AD("../../data/DLPC/151507.h5ad")

cnts = assay(adata, "X")


xy = colData(adata)[, c("array_row", "array_col")]

C = 20
R = 7


BASS <- createBASSObject(list(cnts), list(xy), C = C, R = R, beta_method = "SW", init_method = "mclust", 
                         nsample = 10000)
listAllHyper(BASS)

# reducedDim(adata, "spatial")
# 
# str(adata)
# adata
BASS <- BASS.preprocess(BASS, doLogNormalize = TRUE,
                        geneSelect = "sparkx", nSE = 3000, doPCA = TRUE, 
                        scaleFeature = FALSE, nPC = 20)
