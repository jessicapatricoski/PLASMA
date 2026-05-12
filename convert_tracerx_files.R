library(fst)
library(readr)

setwd("/users/jpatrico/data/lung_new/data/tracerx")

# CLINICAL DATA
clinic <- read_fst("2022-10-14_clinicohistopathological_data.fst")
write.csv(clinic, "clinicohistopathological_data.csv", row.names = FALSE)

patient <- readRDS("20221109_TRACERx421_all_patient_df.rds")
write.csv(as.data.frame(patient), "TRACERx421_all_patient_df.csv", row.names = FALSE)

tumour <- readRDS("20221109_TRACERx421_all_tumour_df.rds")
write.csv(as.data.frame(tumour), "TRACERx421_all_tumour_df.csv", row.names = FALSE)

# MUTATION DATA
mut <- read_fst("20221109_TRACERx421_mutation_table.fst")
write.csv(mut, "TRACERx421_mutation_table.csv", row.names = FALSE)

# EXPRESSION DATA
rsem <- read_fst("2022-10-17_rsem_counts_mat.fst")
write.csv(rsem, "rsem_counts_mat.csv", row.names = FALSE)

rsem_length <- read_fst("2022-10-17_rsem_eff_length_mat.fst")
write.csv(rsem_length, "rsem_eff_length_mat.csv", row.names = FALSE)

rna_purity <- read_fst("19012023_rna_purity_cruk.fst")
write.csv(rna_purity, "rna_purity_cruk.csv", row.names = FALSE)

tpm <- read_fst("2022-10-17_rsem_tpm_mat.fst")
write.csv(tpm, "rsem_tpm_mat.csv", row.names = FALSE)

rna_editing <- read_fst("2022-10-14_rna_editing_table.fst")

all_meta <- read_fst("2022-10-18all_metadata.fst")
